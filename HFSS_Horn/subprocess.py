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

# --- parameter definition ---

def printlog(message):
    """Writes a simple message to the log file."""
    try:
        with open(LOG_PATH, "a") as f:
            f.write(str(message) + "\n")
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

report_name = "S11_Export_Report"
temp_export_path = os.path.join(WATCH_DIR, "temp_hfss_export.csv")


def read_total_length_mm(total_length_path):
    """Read the horn total length and return it as an HFSS millimeter value."""
    with open(total_length_path, "r") as f:
        value = f.read().strip()

    float(value)
    return "{}mm".format(value)

#'''
def runSimulation():
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
                        "1mm",
                        "ZPosition:=",
                        "1mm",
                    ]
                )
            )
            printlog("YZ symmetry boundary face_id resolved by position: {}".format(yz_symmetry_face_id))
            oBoundaryModule.AssignSymmetry(
                [
                    "NAME:Sym1",
                    "Faces:=", [yz_symmetry_face_id],
                    "IsPerfectE:=", True
                ])
            zx_symmetry_face_id = int(
                oEditor.GetFaceByPosition(
                    [
                        "NAME:FaceParameters",
                        "BodyName:=",
                        "Horn",
                        "XPosition:=",
                        "1mm",
                        "YPosition:=",
                        "0mm",
                        "ZPosition:=",
                        "1mm",
                    ]
                )
            )
            printlog("ZX symmetry boundary face_id resolved by position: {}".format(zx_symmetry_face_id))
            oBoundaryModule.AssignSymmetry(
                [
                    "NAME:Sym2",
                    "Faces:=", [zx_symmetry_face_id],
                    "IsPerfectE:=", False
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

            if report_name in oReportModule.GetAllReportNames():
                printlog("[State] Deleting existing report: {}".format(report_name))
                oReportModule.DeleteReports([report_name])

            printlog("[State] Creating report: {}".format(report_name))
            oReportModule.CreateReport(report_name, "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep",
                [
                    "Domain:=", "Sweep"
                ],
                [
                    "Freq:=", ["All"],
                    "a:=", ["Nominal"],
                    "b:=", ["Nominal"],
                    "CenterFreq:=", ["Nominal"],
                    "CoaxOuterDiameter:=", ["Nominal"],
                    "CoaxLength:=", ["Nominal"],
                    "CoaxInnerDiameter:=", ["Nominal"]
                ],
                [
                    "X Component:=", "Freq",
                    "Y Component:=", ["db(mean(mag(S(Port1,Port1))))"]
                ])

            printlog("[State] Exporting report to temporary file: {}".format(temp_export_path))
            oReportModule.ExportToFile(report_name, temp_export_path, False)

    except Exception as e:
        printlog("[ERROR] HFSS simulation: {}".format(e))

    finally:
            # --- 5. Clean up HFSS project for the next run ---
            printlog("[State] Cleaning up a current HFSS simulation...")
            try:
                if oDesign:

                    if report_name in oReportModule.GetAllReportNames():
                        oReportModule.DeleteReports([report_name])

                    oDesign.DeleteFullVariation("All", False)

                # Clean up external imported model
                if oEditor:
                    oEditor.Delete(
                        [
                            "NAME:Selections",
                            "Selections:=", "Horn"
                        ])
                    printlog("[State] Successfully cleaned up Horn")
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
