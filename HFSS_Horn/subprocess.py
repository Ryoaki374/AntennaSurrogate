import os
import time
import json
import itertools
import math
import csv

import ScriptEnv


# Keep these numerical helpers in this file because HFSS IronPython does not
# reliably import the adjacent lib_hfss_metrics module.
C0 = 299792458.0
ELLIPTICITY_FAR_FIELD_SETUP = "Ellipticity Sphere1"
CROSSPOL_FAR_FIELD_SETUP = "Crosspol Sphere1"


def numeric_values(start, stop, step):
    """Return an inclusive floating-point range."""
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


def unique_sorted(values):
    """Return sorted unique floats, tolerating insignificant COM duplicates."""
    result = []
    for value in sorted(float(item) for item in values):
        if not result or abs(value - result[-1]) > 1.0e-10:
            result.append(value)
    return result


def integrate_solid_angle(theta_values, phi_values, grid, component_index):
    """Integrate Gain*abs(sin(theta)) over a theta/phi grid."""
    theta_integrals = []
    for phi_value in phi_values:
        theta_integral = 0.0
        for index in range(1, len(theta_values)):
            theta0_deg = theta_values[index - 1]
            theta1_deg = theta_values[index]
            theta0 = math.radians(theta0_deg)
            theta1 = math.radians(theta1_deg)
            gain0 = grid[(theta0_deg, phi_value)][component_index]
            gain1 = grid[(theta1_deg, phi_value)][component_index]
            y0 = gain0 * abs(math.sin(theta0))
            y1 = gain1 * abs(math.sin(theta1))
            theta_integral += 0.5 * (y0 + y1) * (theta1 - theta0)
        theta_integrals.append(theta_integral)

    result = 0.0
    for index in range(1, len(phi_values)):
        phi0 = math.radians(phi_values[index - 1])
        phi1 = math.radians(phi_values[index])
        result += 0.5 * (
            theta_integrals[index - 1] + theta_integrals[index]
        ) * (phi1 - phi0)
    return result


def _load_retheta_table(path, component, phi_deg=0.0):
    """Load an HFSS Data Table export of re/im(rETheta)."""
    with open(path, "r") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("rETheta export must contain a header and data rows")

    component_name = "re(rETheta)" if component == "re" else "im(rETheta)"
    headers = rows[0]

    # AEDT 2023.2 normally exports Data Tables in this long format:
    # Freq [GHz], Phi [deg], Theta [deg], re(rETheta) [V]
    frequency_column = None
    phi_column = None
    theta_column = None
    component_column = None
    for column_index, header in enumerate(headers):
        header_lower = header.lower()
        if header_lower.startswith("freq"):
            frequency_column = column_index
        elif header_lower.startswith("phi"):
            phi_column = column_index
        elif header_lower.startswith("theta"):
            theta_column = column_index
        if component_name.lower() in header_lower:
            component_column = column_index

    if None in (frequency_column, phi_column, theta_column, component_column):
        raise ValueError("Required rETheta columns were not found in {}".format(path))

    samples_by_frequency = {}
    for row in rows[1:]:
        if abs(float(row[phi_column]) - phi_deg) > 1.0e-10:
            continue
        frequency_ghz = float(row[frequency_column])
        samples_by_frequency.setdefault(frequency_ghz, []).append((
            float(row[theta_column]),
            float(row[component_column]),
        ))
    if not samples_by_frequency:
        raise ValueError("No {} rows found in {}".format(component_name, path))

    theta = None
    data = {}
    for frequency_ghz in sorted(samples_by_frequency):
        samples = sorted(samples_by_frequency[frequency_ghz])
        current_theta = [sample[0] for sample in samples]
        if theta is None:
            theta = current_theta
        elif current_theta != theta:
            raise ValueError("Theta samples differ between frequencies in {}".format(path))
        data[frequency_ghz] = [sample[1] for sample in samples]
    return theta, data


def _unwrap(phases):
    """Unwrap radians using the pi discontinuity convention."""
    if not phases:
        return []
    result = [phases[0]]
    offset = 0.0
    for index in range(1, len(phases)):
        delta = phases[index] - phases[index - 1]
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        result.append(phases[index] + offset)
    return result


def calculate_phase_centers(
    re_path,
    im_path,
    theta_min_deg=-10.0,
    theta_max_deg=10.0,
    z_min_mm=-30.0,
    z_max_mm=30.0,
    dz_mm=0.2,
    phase_sign=-1.0,
):
    """Find the z that minimizes phase peak-to-peak at every frequency."""
    theta_re, re_data = _load_retheta_table(re_path, "re")
    theta_im, im_data = _load_retheta_table(im_path, "im")
    if theta_re != theta_im:
        raise ValueError("Theta samples in re/im rETheta exports do not match")

    selected = [
        index for index, theta in enumerate(theta_re)
        if theta_min_deg <= theta <= theta_max_deg
    ]
    selected.sort(key=lambda index: theta_re[index])
    if len(selected) < 3:
        raise ValueError("Too few theta samples in the phase-center range")

    theta_rad = [math.radians(theta_re[index]) for index in selected]
    z_candidates_mm = numeric_values(z_min_mm, z_max_mm, dz_mm)
    frequencies = sorted(set(re_data).intersection(im_data))
    results = []

    for frequency_ghz in frequencies:
        phase0 = _unwrap([
            math.atan2(im_data[frequency_ghz][index], re_data[frequency_ghz][index])
            for index in selected
        ])
        wave_number = 2.0 * math.pi * frequency_ghz * 1.0e9 / C0

        best_z_mm = None
        best_pkpk_rad = None
        for z_mm in z_candidates_mm:
            z_m = z_mm * 1.0e-3
            corrected_phase = [
                phase + phase_sign * wave_number * z_m * math.cos(theta)
                for phase, theta in zip(phase0, theta_rad)
            ]
            phase_pkpk_rad = max(corrected_phase) - min(corrected_phase)
            if best_pkpk_rad is None or phase_pkpk_rad < best_pkpk_rad:
                best_z_mm = z_mm
                best_pkpk_rad = phase_pkpk_rad

        results.append((
            frequency_ghz,
            best_z_mm,
            math.degrees(best_pkpk_rad),
        ))
    return results

# --- Initialize the Scripting Environment ---
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")

# --- Configuration & Global Constants ---
LOG_PATH = r"T:\RAkizawa\HFSS_Horn\src\output_log.log"
CONFIG_PATH = r'T:\RAkizawa\HFSS_Horn\src\_config_HFSS.json'
TOTAL_LENGTH_FILENAME = '.total_length'
SCRIPT_START_TIME = time.time()

# --- parameter definition ---

def _elapsed_time_label():
    """Return elapsed runtime as a [minutes:seconds] log prefix."""
    elapsed_seconds = int(time.time() - SCRIPT_START_TIME)
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    return "[{}:{:02d}]".format(minutes, seconds)

def printlog(message):
    """Writes a timestamped message to the log file."""
    try:
        with open(LOG_PATH, "a") as f:
            f.write("{} {}\n".format(_elapsed_time_label(), str(message)))
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write("[ERROR][printlog] {}".format(str(e)))

# Clear the log file at the start of the script for a clean debug session
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)
printlog("--- HFSS Subroutine Script Initialized ---")

# --- Load Settings from Config File ---
try:
    printlog("Loading configuration from: {}".format(CONFIG_PATH))
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    WATCH_DIR = config['WATCH_DIR']
    INPUT_FILE = config['INPUT_FILE']
    MODEL_FILE = config['MODEL_FILE']
    RESULTS_FILE = config['RESULTS_FILE']
    TEMP_OUTPUTS = config.get('TEMP_OUTPUTS', [])
    #PARAM_KEYS = config['param_names']
    #printlog("[Debug] {}, {}".format(config["n_repeats"], config["n_simulation"]))

    DONE_FLAG_FILE = config.get("DONE_FLAG_FILE", os.path.join(WATCH_DIR, "hfss.done"))
    RESULT_READY_FILE = config.get(
        "RESULT_READY_FILE", os.path.join(WATCH_DIR, "hfss_result.ready")
    )
    printlog("Configuration loaded. WATCH_DIR: {}. Done flag: {}".format(WATCH_DIR, DONE_FLAG_FILE))
except Exception as e:
    printlog("[ERROR][loading config] {}".format(e))
    exit()

# Create the folder if it does not exist
if not os.path.exists(WATCH_DIR):
    printlog("[ERROR][Watching dir] Creating: {}".format(WATCH_DIR))
    os.makedirs(WATCH_DIR)

# --- HFSS Object Initialization ---
try:
    oProject = oDesktop.GetActiveProject()
    oDesign = oProject.GetActiveDesign()
    oOptiModule = oDesign.GetModule("Optimetrics")
    oReportModule = oDesign.GetModule("ReportSetup")
    printlog("HFSS Objects Initialized: Project='{}', Design='{}'".format(oProject.GetName(), oDesign.GetName()))
except AttributeError:
    printlog("[ERROR][HFSS_init] Could not get active Project or Design.")
    exit()

temp_output_paths = {}
for output in TEMP_OUTPUTS:
    if output.get("name") and output.get("path"):
        temp_output_paths[output["name"]] = output["path"]

temp_output_paths.setdefault("S11", os.path.join(WATCH_DIR, "temp_S11_export.csv"))

for output_name in sorted(temp_output_paths):
    printlog(
        "[Config] {} output path: {}".format(
            output_name, temp_output_paths[output_name]
        )
    )


REPORT_SPECS = [
    {
        "output_name": "S11",
        "report_name": "S11_Export_Report",
        "category": "Modal Solution Data",
        "context": ["Domain:=", "Sweep"],
        "families": [
            "Freq:=", ["All"],
            "a:=", ["Nominal"],
            "b:=", ["Nominal"],
            "CenterFreq:=", ["Nominal"],
            "CoaxOuterDiameter:=", ["Nominal"],
            "CoaxLength:=", ["Nominal"],
            "CoaxInnerDiameter:=", ["Nominal"],
        ],
        "y_component": "db(max(mag(S(Port1:1,Port1:1))))",
    },
]

PHASE_REPORT_SPECS = [
    {
        "report_name": "rETheta_Real_Export_Report",
        "filename": "temp_rerETheta_export.csv",
        "y_component": "re(rETheta)",
    },
    {
        "report_name": "rETheta_Imag_Export_Report",
        "filename": "temp_imrETheta_export.csv",
        "y_component": "im(rETheta)",
    },
]


def export_reports():
    """Create and export the configured native HFSS reports."""
    existing_reports = oReportModule.GetAllReportNames()
    for report in REPORT_SPECS:
        output_path = temp_output_paths.get(report["output_name"])
        if not output_path:
            printlog("[State] Skipping unconfigured output: {}".format(report["output_name"]))
            continue

        report_name = report["report_name"]
        if report_name in existing_reports:
            printlog("[State] Deleting existing report: {}".format(report_name))
            oReportModule.DeleteReports([report_name])

        printlog("[State] Creating report: {}".format(report_name))
        oReportModule.CreateReport(
            report_name,
            report["category"],
            "Rectangular Plot",
            "Setup1 : Sweep",
            report["context"],
            report["families"],
            ["X Component:=", "Freq", "Y Component:=", [report["y_component"]]],
        )
        printlog("[State] Exporting {} to: {}".format(report_name, output_path))
        oReportModule.ExportToFile(report_name, output_path, False)
        for _ in range(50):
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                "{} did not create a non-empty file at {}".format(
                    report_name, output_path
                )
            )
        printlog(
            "[State] Verified {} output: {} bytes at {}".format(
                report_name, os.path.getsize(output_path), output_path
            )
        )


def _write_rows(output_path, header, rows):
    """Publish a completed CSV without exposing a partially written result."""
    partial_path = output_path + ".partial"
    with open(partial_path, "w") as output_file:
        output_file.write(",".join(str(value) for value in header) + "\n")
        for row in rows:
            output_file.write(",".join("{:.16g}".format(value) for value in row) + "\n")
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(partial_path, output_path)


def publish_result_ready():
    """Notify the main process only after every requested output is complete."""
    partial_path = RESULT_READY_FILE + ".partial"
    with open(partial_path, "w") as ready_file:
        ready_file.write("ready\n")
    if os.path.exists(RESULT_READY_FILE):
        os.remove(RESULT_READY_FILE)
    os.rename(partial_path, RESULT_READY_FILE)
    printlog("[State] Published result-ready flag: {}".format(RESULT_READY_FILE))


def _find_name_case_insensitive(names, requested):
    requested_lower = requested.lower()
    for name in names:
        if str(name).lower() == requested_lower:
            return str(name)
    raise RuntimeError("Sweep '{}' was not returned".format(requested))


def export_ellipticity_field():
    """Export complex rEL3X samples used for FFT beam ellipticity fitting."""
    output_path = temp_output_paths.get("ellipticity")
    if not output_path:
        printlog("[State] Skipping unconfigured output: ellipticity")
        return

    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    frequency_values = numeric_values(80.0, 175.0, 5.0)
    try:
        with open(partial_path, "wb") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "Freq [GHz]",
                "Theta [deg]",
                "Phi [deg]",
                "re(rEL3X) [V]",
                "im(rEL3X) [V]",
            ])

            for index, frequency_ghz in enumerate(frequency_values, 1):
                frequency = "{:g}GHz".format(frequency_ghz)
                result_array = oReportModule.GetSolutionDataPerVariation(
                    "Far Fields",
                    "Setup1 : Sweep",
                    ["Context:=", ELLIPTICITY_FAR_FIELD_SETUP],
                    [
                        "Theta:=", ["All"],
                        "Phi:=", ["All"],
                        "Freq:=", [frequency],
                    ],
                    ["rEL3X"],
                )
                if result_array is None or len(result_array) != 1:
                    raise RuntimeError(
                        "Unexpected rEL3X result count at {}".format(frequency)
                    )

                data = result_array[0]
                try:
                    sweep_order = [str(name) for name in list(data.GetSweepNames())]
                    sweep_order.reverse()
                    theta_name = _find_name_case_insensitive(sweep_order, "Theta")
                    phi_name = _find_name_case_insensitive(sweep_order, "Phi")
                    frequency_name = _find_name_case_insensitive(sweep_order, "Freq")

                    sweep_values = {}
                    sweep_units = {}
                    for sweep_name in sweep_order:
                        sweep_values[sweep_name] = unique_sorted(
                            data.GetSweepValues(sweep_name, False)
                        )
                        sweep_units[sweep_name] = str(
                            data.GetSweepUnits(sweep_name)
                        ).lower()

                    dimensions = [sweep_values[name] for name in sweep_order]
                    expected_count = 1
                    for dimension in dimensions:
                        expected_count *= len(dimension)

                    real_values = [
                        float(value)
                        for value in data.GetRealDataValues("rEL3X", False)
                    ]
                    imag_values = [
                        float(value)
                        for value in data.GetImagDataValues("rEL3X", False)
                    ]
                    if len(real_values) != expected_count or len(imag_values) != expected_count:
                        raise RuntimeError(
                            "rEL3X returned an unexpected value count at {}".format(
                                frequency
                            )
                        )

                    for flat_index, coordinates in enumerate(
                        itertools.product(*dimensions)
                    ):
                        point = dict(zip(sweep_order, coordinates))
                        returned_frequency = float(point[frequency_name])
                        theta_deg = float(point[theta_name])
                        phi_deg = float(point[phi_name])

                        frequency_unit = sweep_units[frequency_name]
                        if frequency_unit == "hz":
                            returned_frequency /= 1.0e9
                        elif frequency_unit == "khz":
                            returned_frequency /= 1.0e6
                        elif frequency_unit == "mhz":
                            returned_frequency /= 1.0e3
                        if "rad" in sweep_units[theta_name]:
                            theta_deg = math.degrees(theta_deg)
                        if "rad" in sweep_units[phi_name]:
                            phi_deg = math.degrees(phi_deg)

                        writer.writerow([
                            returned_frequency,
                            theta_deg,
                            phi_deg,
                            real_values[flat_index],
                            imag_values[flat_index],
                        ])
                finally:
                    try:
                        data.ReleaseData()
                    except Exception:
                        pass

                printlog(
                    "[Ellipticity {}/{}] Exported rEL3X at {}".format(
                        index, len(frequency_values), frequency
                    )
                )

        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(partial_path, output_path)
        printlog("[State] Exported ellipticity rEL3X field to: {}".format(output_path))
    except Exception:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise


def _get_far_field_grid(frequency, theta_values, phi_values):
    """Read the GainL3 grid used by the verified Crosspol integration."""
    expressions = ["GainL3Y", "GainTotal"]
    result_array = oReportModule.GetSolutionDataPerVariation(
        "Far Fields",
        "Setup1 : Sweep",
        ["Context:=", CROSSPOL_FAR_FIELD_SETUP],
        [
            "Theta:=", ["All"],
            "Phi:=", ["All"],
            "Freq:=", [frequency],
        ],
        expressions,
    )
    if result_array is None or len(result_array) != 1:
        raise RuntimeError("Unexpected far-field result count at {}".format(frequency))

    data = result_array[0]
    try:
        sweep_order = [str(name) for name in list(data.GetSweepNames())]
        sweep_order.reverse()
        theta_name = _find_name_case_insensitive(sweep_order, "Theta")
        phi_name = _find_name_case_insensitive(sweep_order, "Phi")

        sweep_values = {}
        sweep_units = {}
        for sweep_name in sweep_order:
            sweep_values[sweep_name] = unique_sorted(
                data.GetSweepValues(sweep_name, False)
            )
            try:
                sweep_units[sweep_name] = str(data.GetSweepUnits(sweep_name)).lower()
            except Exception:
                sweep_units[sweep_name] = ""

        dimensions = [sweep_values[name] for name in sweep_order]
        expected_count = 1
        for dimension in dimensions:
            expected_count *= len(dimension)

        expression_values = {}
        for expression in expressions:
            values = [float(value) for value in data.GetRealDataValues(expression, False)]
            if len(values) != expected_count:
                raise RuntimeError("{} returned an unexpected value count".format(expression))
            expression_values[expression] = values

        wanted_theta = dict((round(value, 8), value) for value in theta_values)
        wanted_phi = dict((round(value, 8), value) for value in phi_values)
        grid = {}
        for flat_index, coordinates in enumerate(itertools.product(*dimensions)):
            coordinate = dict(zip(sweep_order, coordinates))
            theta_deg = float(coordinate[theta_name])
            phi_deg = float(coordinate[phi_name])
            if "rad" in sweep_units[theta_name]:
                theta_deg = math.degrees(theta_deg)
            if "rad" in sweep_units[phi_name]:
                phi_deg = math.degrees(phi_deg)

            theta_key = round(theta_deg, 8)
            phi_key = round(phi_deg, 8)
            if theta_key not in wanted_theta or phi_key not in wanted_phi:
                continue

            theta_value = wanted_theta[theta_key]
            phi_value = wanted_phi[phi_key]
            gains = []
            for expression in expressions:
                value = expression_values[expression][flat_index]
                if math.isnan(value) or math.isinf(value):
                    if abs(math.sin(math.radians(theta_value))) < 1.0e-14:
                        value = 0.0
                    else:
                        raise RuntimeError(
                            "Non-finite {} at {}, Phi={}deg, Theta={}deg".format(
                                expression, frequency, phi_value, theta_value
                            )
                        )
                gains.append(value)
            grid[(theta_value, phi_value)] = tuple(gains)

        expected_grid_size = len(theta_values) * len(phi_values)
        if len(grid) != expected_grid_size:
            raise RuntimeError(
                "Incomplete Crosspol angular grid at {}: returned {}, expected {}".format(
                    frequency, len(grid), expected_grid_size
                )
            )
        return grid
    finally:
        try:
            data.ReleaseData()
        except Exception:
            pass


def export_crosspol():
    """Calculate Crosspol from 80 to 175 GHz at 5 GHz intervals."""
    output_path = temp_output_paths.get("Crosspol")
    if not output_path:
        printlog("[State] Skipping unconfigured output: Crosspol")
        return

    frequency_values = numeric_values(80.0, 175.0, 5.0)
    theta_values = numeric_values(-15.0, 15.0, 0.5)
    phi_values = numeric_values(0.0, 90.0, 1.0)
    rows = []
    printlog("[State] Calculating Crosspol over 80-175 GHz in 5 GHz steps")
    for index, frequency_ghz in enumerate(frequency_values, 1):
        frequency = "{:g}GHz".format(frequency_ghz)
        grid = _get_far_field_grid(frequency, theta_values, phi_values)
        integral_l3y = integrate_solid_angle(theta_values, phi_values, grid, 0)
        integral_total = integrate_solid_angle(theta_values, phi_values, grid, 1)
        if integral_total == 0.0:
            raise ZeroDivisionError("Integrated GainTotal is zero at {}".format(frequency))
        crosspol = integral_l3y / integral_total
        rows.append((frequency_ghz, crosspol))
        printlog(
            "[Crosspol {}/{}] {}: Crosspol={:.16g}".format(
                index, len(frequency_values), frequency, crosspol
            )
        )

    _write_rows(output_path, ["Frequency_GHz", "Crosspol"], rows)
    printlog("[State] Exported Crosspol to: {}".format(output_path))


def export_phasecenter():
    """Export complex rETheta and calculate phase center versus frequency."""
    output_path = temp_output_paths.get("phasecenter")
    if not output_path:
        printlog("[State] Skipping unconfigured output: phasecenter")
        return

    existing_reports = oReportModule.GetAllReportNames()
    raw_paths = []
    for report in PHASE_REPORT_SPECS:
        report_name = report["report_name"]
        raw_path = os.path.join(WATCH_DIR, report["filename"])
        raw_paths.append(raw_path)
        if report_name in existing_reports:
            oReportModule.DeleteReports([report_name])
        if os.path.exists(raw_path):
            os.remove(raw_path)

        printlog("[State] Creating report: {}".format(report_name))
        oReportModule.CreateReport(
            report_name,
            "Far Fields",
            "Data Table",
            "Setup1 : Sweep",
            ["Context:=", "Infinite Sphere1"],
            [
                "Theta:=", ["All"],
                "Phi:=", ["0deg"],
                "Freq:=", ["All"],
            ],
            [
                "X Component:=", "Theta",
                "Y Component:=", [report["y_component"]],
            ],
        )
        oReportModule.ExportToFile(report_name, raw_path, False)
        printlog("[State] Exported {} to: {}".format(report_name, raw_path))

    results = calculate_phase_centers(raw_paths[0], raw_paths[1])
    _write_rows(
        output_path,
        ["Frequency_GHz", "PhaseCenterZ_mm", "MinimumPhasePkPk_deg"],
        results,
    )
    printlog("[State] Exported phasecenter to: {}".format(output_path))

    for raw_path in raw_paths:
        try:
            os.remove(raw_path)
        except OSError:
            pass


def read_total_length_mm(total_length_path):
    """Read the horn total length and return it as an HFSS millimeter value."""
    with open(total_length_path, "r") as f:
        value = f.read().strip()

    float(value)
    return "{}mm".format(value)

#'''
def runSimulation():
    oRadFieldModule = None
    try:
            if os.path.exists(RESULT_READY_FILE):
                os.remove(RESULT_READY_FILE)
            ready_partial_path = RESULT_READY_FILE + ".partial"
            if os.path.exists(ready_partial_path):
                os.remove(ready_partial_path)

            # model import
            printlog("[State] Importing step file from: {}".format(MODEL_FILE))
            oEditor = oDesign.SetActiveEditor("3D Modeler")
            oEditor.Import(
                [
                    "NAME:NativeBodyParameters",
                    "HealOption:=", 0,
                    "Options:=", "0",
                    "FileType:=", "UnRecognized",
                    "MaxStitchTol:=", -1,
                    "ImportFreeSurfaces:=", False,
                    "GroupByAssembly:=", False,
                    "CreateGroup:=", True,
                    "STLFileUnit:=", "mm",
                    "MergeFacesAngle:=", -1,
                    "HealSTL:=", True,
                    "ReduceSTL:=", False,
                    "ReduceMaxError:=", 0,
                    "ReducePercentage:=", 100,
                    "PointCoincidenceTol:=", 1E-08,
                    "CreateLightweightPart:=", False,
                    "ImportMaterialNames:=", False,
                    "SeparateDisjointLumps:=", False,
                    "SourceFile:=", MODEL_FILE[0]
                ])
            oEditor = oDesign.SetActiveEditor("3D Modeler")
            oEditor.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:Geometry3DAttributeTab",
                        [
                            "NAME:PropServers",
                            "OpenCASCADESTEPtranslator7"
                        ],
                        [
                            "NAME:ChangedProps",
                            [
                                "NAME:Name",
                                "Value:=", "Horn"
                            ]
                        ]
                    ]
                ])
            oEditor = oDesign.SetActiveEditor("3D Modeler")
            oEditor.AssignMaterial(
                [
                    "NAME:Selections",
                    "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                    "AllowRegionSelectionForPMLCreation:=", True,
                    "Selections:=", "Horn"
                ],
                [
                    "NAME:Attributes",
                    "MaterialValue:=", "\"vacuum\"",
                    "SolveInside:=", True,
                    "ShellElement:=", False,
                    "ShellElementThickness:=", "nan ",
                    "ReferenceTemperature:=", "nan ",
                    "IsMaterialEditable:=", True,
                    "UseMaterialAppearance:=", False,
                    "IsLightweight:=", False
                ])

            # boundary assignment
            total_length_path = os.path.join(WATCH_DIR, TOTAL_LENGTH_FILENAME)
            z_position = read_total_length_mm(total_length_path)
            printlog("ZPosition loaded from {}: {}".format(total_length_path, z_position))
            face_id = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "Horn",
                        "XPosition:=",
                        "0mm",
                        "YPosition:=",
                        "0mm",
                        "ZPosition:=",
                        z_position,
                    ]
                )
            )
            printlog("Radiation boundary face_id resolved from .total_length: {}".format(face_id))
            oBoundaryModule = oDesign.GetModule("BoundarySetup")
            oBoundaryModule.AssignRadiation(
                [
                    "NAME:Rad1",
                    "Faces:=", [face_id]
                ])

            # split Horn into the positive YZ/ZX half model before assigning symmetry boundaries
            oEditor = oDesign.SetActiveEditor("3D Modeler")
            oEditor.Split(
                [
                    "NAME:Selections",
                    "Selections:=", "Horn",
                    "NewPartsModelFlag:=", "Model"
                ],
                [
                    "NAME:SplitToParameters",
                    "SplitPlane:=", "YZ",
                    "WhichSide:=", "PositiveOnly",
                    "ToolType:=", "PlaneTool",
                    "ToolEntityID:=", -1,
                    "SplitCrossingObjectsOnly:=", False,
                    "DeleteInvalidObjects:=", True
                ])
            oEditor.Split(
                [
                    "NAME:Selections",
                    "Selections:=", "Horn",
                    "NewPartsModelFlag:=", "Model"
                ],
                [
                    "NAME:SplitToParameters",
                    "SplitPlane:=", "ZX",
                    "WhichSide:=", "PositiveOnly",
                    "ToolType:=", "PlaneTool",
                    "ToolEntityID:=", -1,
                    "SplitCrossingObjectsOnly:=", False,
                    "DeleteInvalidObjects:=", True
                ])

            yz_symmetry_face_id = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "Horn",
                        "XPosition:=",
                        "0mm",
                        "YPosition:=",
                        "0.5mm",
                        "ZPosition:=",
                        "1mm",
                    ]
                )
            )
            yz_symmetry_face_id_WG = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "WG",
                        "XPosition:=",
                        "0mm",
                        "YPosition:=",
                        "0.5mm",
                        "ZPosition:=",
                        "-1mm",
                    ]
                )
            )
            
            printlog("YZ symmetry boundary face_id resolved by position: [{}, {}]".format(yz_symmetry_face_id, yz_symmetry_face_id_WG))
            oBoundaryModule.AssignSymmetry(
                [
                    "NAME:Sym1",
                    "Faces:=", [yz_symmetry_face_id, yz_symmetry_face_id_WG],
                    "IsPerfectE:=", True
                ])
            zx_symmetry_face_id = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "Horn",
                        "XPosition:=",
                        "0.5mm",
                        "YPosition:=",
                        "0mm",
                        "ZPosition:=",
                        "1mm",
                    ]
                )
            )
            zx_symmetry_face_id_WG = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "WG",
                        "XPosition:=",
                        "0.5mm",
                        "YPosition:=",
                        "0mm",
                        "ZPosition:=",
                        "-1mm",
                    ]
                )
            )
            printlog("ZX symmetry boundary face_id resolved by position: {}".format(zx_symmetry_face_id))
            oBoundaryModule.AssignSymmetry(
                [
                    "NAME:Sym2",
                    "Faces:=", [zx_symmetry_face_id, zx_symmetry_face_id_WG],
                    "IsPerfectE:=", False
                ])

            oRadFieldModule = oDesign.GetModule("RadField")
            if "Infinite Sphere1" in oRadFieldModule.GetChildNames():
                printlog("[State] Deleting existing far-field setup: Infinite Sphere1")
                oRadFieldModule.DeleteSetup(["Infinite Sphere1"])
            oRadFieldModule.InsertInfiniteSphereSetup(
                [
                    "NAME:Infinite Sphere1",
                    "UseCustomRadiationSurface:=", False,
                    "CSDefinition:=", "Theta-Phi",
                    "Polarization:=", "Linear",
                    "ThetaStart:=", "-30deg",
                    "ThetaStop:=", "30deg",
                    "ThetaStep:=", "0.1deg",
                    "PhiStart:=", "0deg",
                    "PhiStop:=", "90deg",
                    "PhiStep:=", "1deg",
                    "UseLocalCS:=", False,
                ])
            if ELLIPTICITY_FAR_FIELD_SETUP in oRadFieldModule.GetChildNames():
                printlog(
                    "[State] Deleting existing far-field setup: {}".format(
                        ELLIPTICITY_FAR_FIELD_SETUP
                    )
                )
                oRadFieldModule.DeleteSetup([ELLIPTICITY_FAR_FIELD_SETUP])
            oRadFieldModule.InsertInfiniteSphereSetup(
                [
                    "NAME:" + ELLIPTICITY_FAR_FIELD_SETUP,
                    "UseCustomRadiationSurface:=", False,
                    "CSDefinition:=", "Theta-Phi",
                    "Polarization:=", "Linear",
                    "ThetaStart:=", "0deg",
                    "ThetaStop:=", "9.5deg",
                    "ThetaStep:=", "0.1deg",
                    "PhiStart:=", "0deg",
                    "PhiStop:=", "90deg",
                    "PhiStep:=", "1deg",
                    "UseLocalCS:=", False,
                ]
            )
            if CROSSPOL_FAR_FIELD_SETUP in oRadFieldModule.GetChildNames():
                printlog(
                    "[State] Deleting existing far-field setup: {}".format(
                        CROSSPOL_FAR_FIELD_SETUP
                    )
                )
                oRadFieldModule.DeleteSetup([CROSSPOL_FAR_FIELD_SETUP])
            oRadFieldModule.InsertInfiniteSphereSetup(
                [
                    "NAME:" + CROSSPOL_FAR_FIELD_SETUP,
                    "UseCustomRadiationSurface:=", False,
                    "CSDefinition:=", "Theta-Phi",
                    "Polarization:=", "Linear",
                    "ThetaStart:=", "-15deg",
                    "ThetaStop:=", "15deg",
                    "ThetaStep:=", "0.5deg",
                    "PhiStart:=", "0deg",
                    "PhiStop:=", "90deg",
                    "PhiStep:=", "1deg",
                    "UseLocalCS:=", False,
                ]
            )

            oProject.Save()

            # remove imported models
            if os.path.exists(MODEL_FILE[0]):
                try:
                    os.remove(MODEL_FILE[0])
                except:
                    printlog("[ERROR] Could not delete input file.")

            #Validation
            try:
               check = oDesign.ValidateDesign()
               if check == 1:
                   printlog("Design validated successfully.")
               else:
                   printlog("Design validation failed.")
            except:
               printlog("[ERROR] Design validation failed with an exception.")

            # solve
            oDesign.Analyze("Setup1 : Sweep")
            printlog("[State] Solve complete.")

            # setup
            oReportModule = oDesign.GetModule("ReportSetup")

            export_reports()
            export_ellipticity_field()
            export_phasecenter()
            export_crosspol()
            publish_result_ready()

    except Exception as e:
        printlog("[ERROR] HFSS simulation: {}".format(e))

    finally:
            # --- 5. Clean up HFSS project for the next run ---
            printlog("[State] Cleaning up a current HFSS simulation...")
            try:
                if oDesign:

                    existing_reports = oReportModule.GetAllReportNames()
                    reports_to_delete = [
                        report["report_name"] for report in REPORT_SPECS + PHASE_REPORT_SPECS
                        if report["report_name"] in existing_reports
                    ]
                    if reports_to_delete:
                        oReportModule.DeleteReports(reports_to_delete)

                    if oRadFieldModule:
                        oRadFieldModule.DeleteSetup(
                            [
                                "Infinite Sphere1",
                                ELLIPTICITY_FAR_FIELD_SETUP,
                                CROSSPOL_FAR_FIELD_SETUP,
                            ]
                        )

                    oDesign.DeleteFullVariation("All", False)

                # Clean up external imported model
                if oEditor:
                    oEditor.Delete(
                        [
                            "NAME:Selections",
                            "Selections:=", "Horn"
                        ])
                    printlog("[State] Successfully cleaned up Horn")
                if oBoundaryModule:
                    oBoundaryModule.DeleteAllBoundaries()
                    
            except Exception as cleanup_e:
                printlog("[ERROR] HFSS object cleanup: {}".format(cleanup_e))


# --- Main Loop ---
printlog("[State] Entering main loop...")

while True:
    if os.path.exists(DONE_FLAG_FILE):
        printlog("[State] Done flag detected. Exiting subprocess loop.")
        break

    if os.path.exists(MODEL_FILE[0]):
        printlog("[State] Detected model file. Starting simulation run.")

        time.sleep(0.2)

        # 2. Run Simulation
        runSimulation()

        if os.path.exists(DONE_FLAG_FILE):
            printlog("[State] Done flag detected after simulation run.")
            break

    time.sleep(1)

printlog("--- All Completed ---")

#'''



