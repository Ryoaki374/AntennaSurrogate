import os
import time
import csv
import json

import ScriptEnv

# --- Initialize the Scripting Environment ---
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")

# --- Configuration & Global Constants ---
LOG_PATH = r"T:\RAkizawa\HFSS_Horn\src\output_log.txt"
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
        "output_name": "XPD",
        "report_name": "XPD_Export_Report",
        "category": "Antenna Parameters",
        "context": ["Context:=", "Infinite Sphere1"],
        "families": ["Freq:=", ["All"]],
        "y_component": "dB20(MaxrELudwig3YComp/MaxrELudwig3XComp)",
    },
    {
        "output_name": "ellipticity",
        "report_name": "Ellipticity_Export_Report",
        "category": "Antenna Parameters",
        "context": ["Context:=", "Infinite Sphere1"],
        "families": ["Freq:=", ["All"]],
        "y_component": "dB(AxialRatioValue)",
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

    except Exception as e:
        printlog("[ERROR] HFSS simulation: {}".format(e))

    finally:
            # --- 5. Clean up HFSS project for the next run ---
            printlog("[State] Cleaning up a current HFSS simulation...")
            try:
                if oDesign:

                    existing_reports = oReportModule.GetAllReportNames()
                    reports_to_delete = [
                        report["report_name"] for report in REPORT_SPECS
                        if report["report_name"] in existing_reports
                    ]
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
