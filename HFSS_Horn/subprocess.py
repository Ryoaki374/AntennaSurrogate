import os
import time
import json
import itertools
import math

import ScriptEnv

from lib_hfss_metrics import (
    calculate_phase_centers,
    integrate_solid_angle,
    numeric_values,
    unique_sorted,
)

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

temp_output_paths.setdefault("S11", os.path.join(WATCH_DIR, "temp_hfss_export.csv"))


def phase_center_imag_path(real_path):
    """Return the private imaginary-field export paired with a real-field CSV."""
    root, extension = os.path.splitext(real_path)
    return root + "_imag" + extension

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
        "y_component": "db(mean(mag(S(Port1,Port1))))",
    },
    {
        "output_name": "ellipticity",
        "report_name": "Ellipticity_Export_Report",
        "category": "Far Fields",
        "context": ["Context:=", "Infinite Sphere1"],
        "families": [
            "Theta:=", ["All"],
            "Freq:=", ["All"],
            "Phi:=", ["0deg", "90deg"],
        ],
        "y_component": "XWidthAtYVal(GainTotal/PeakGain, 0.5)",
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
    """Create and export each configured scalar-output report."""
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

    export_phase_center_reports(existing_reports)


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


def _find_name_case_insensitive(names, requested):
    requested_lower = requested.lower()
    for name in names:
        if str(name).lower() == requested_lower:
            return str(name)
    raise RuntimeError("Sweep '{}' was not returned".format(requested))


def _get_far_field_grid(frequency, theta_values, phi_values):
    """Read the GainL3 grid used by the verified Crosspol integration."""
    expressions = ["GainL3X", "GainL3Y", "GainTotal"]
    result_array = oReportModule.GetSolutionDataPerVariation(
        "Far Fields",
        "Setup1 : Sweep",
        ["Context:=", "Infinite Sphere1"],
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
    """Calculate the band Crosspol samples using the attached test.py method."""
    output_path = temp_output_paths.get("Crosspol")
    if not output_path:
        printlog("[State] Skipping unconfigured output: Crosspol")
        return

    frequency_values = numeric_values(85.0, 175.0, 1.0)
    theta_values = numeric_values(-15.0, 15.0, 0.5)
    phi_values = numeric_values(0.0, 90.0, 1.0)
    rows = []
    printlog("[State] Calculating Crosspol over 85-175 GHz")
    for index, frequency_ghz in enumerate(frequency_values, 1):
        frequency = "{:g}GHz".format(frequency_ghz)
        grid = _get_far_field_grid(frequency, theta_values, phi_values)
        integral_l3x = integrate_solid_angle(theta_values, phi_values, grid, 0)
        integral_l3y = integrate_solid_angle(theta_values, phi_values, grid, 1)
        integral_total = integrate_solid_angle(theta_values, phi_values, grid, 2)
        if integral_total == 0.0:
            raise ZeroDivisionError("Integrated GainTotal is zero at {}".format(frequency))
        copol = integral_l3x / integral_total
        crosspol = integral_l3y / integral_total
        rows.append((frequency_ghz, crosspol))
        printlog(
            "[Crosspol {}/{}] {}: Copol={:.16g}, Crosspol={:.16g}".format(
                index, len(frequency_values), frequency, copol, crosspol
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
            export_phasecenter()
            export_crosspol()

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
                    reports_to_delete.extend(
                        report["report_name"] for report in PHASE_CENTER_REPORTS
                        if report["report_name"] in existing_reports
                    )
                    if reports_to_delete:
                        oReportModule.DeleteReports(reports_to_delete)

                    if oRadFieldModule:
                        oRadFieldModule.DeleteSetup(["Infinite Sphere1"])

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

