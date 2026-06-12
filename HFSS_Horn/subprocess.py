# HFSS-side subprocess script for horn-only STEP import and simulation.
# This file is executed inside the HFSS Python environment.

import os
import time
import json


def printlog(message):
    print(message)


try:
    config_path = os.path.join(os.getcwd(), "_config_HFSS.json")
    if not os.path.exists(config_path):
        config_path = "_config_HFSS.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    WATCH_DIR = config["WATCH_DIR"]
    MODEL_FILE = config["MODEL_FILE"]
    RESULTS_FILE = config["RESULTS_FILE"]
    DONE_FLAG_FILE = config.get("DONE_FLAG_FILE", os.path.join(WATCH_DIR, "hfss.done"))
    printlog("Configuration loaded. WATCH_DIR: {}. Done flag: {}".format(WATCH_DIR, DONE_FLAG_FILE))
except Exception as e:
    printlog("[ERROR][loading config] {}".format(e))
    exit()

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
HORN_OBJECT_NAME = "Horn"


def _all_model_files_ready(model_files):
    for model_file in model_files:
        if not os.path.exists(model_file):
            return False
        if os.path.getsize(model_file) <= 0:
            return False
    return True


def _remove_model_files(model_files):
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                printlog("[State] Deleted model file: {}".format(model_file))
            except Exception as e:
                printlog("[ERROR] Could not delete model file {}: {}".format(model_file, e))


def runSimulation():
    oEditor = None
    try:
        if len(MODEL_FILE) != 1:
            raise ValueError("Horn workflow expects exactly one STEP model, got {}".format(len(MODEL_FILE)))

        printlog("[State] Importing horn STEP file from: {}".format(MODEL_FILE[0]))
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
                            "Value:=", HORN_OBJECT_NAME
                        ]
                    ]
                ]
            ])

        # The generated horn is a vacuum volume. Keep SolveInside enabled so the
        # imported solid participates as the dielectric/air region intended by
        # the CAD generator.
        oEditor.AssignMaterial(
            [
                "NAME:Selections",
                "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                "AllowRegionSelectionForPMLCreation:=", True,
                "Selections:=", HORN_OBJECT_NAME
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

        oProject.Save()
        _remove_model_files(MODEL_FILE)

        try:
            check = oDesign.ValidateDesign()
            if check == 1:
                printlog("Design validated successfully.")
            else:
                printlog("Design validation failed.")
        except Exception:
            printlog("[ERROR] Design validation failed with an exception.")

        oDesign.Analyze("Setup1 : Sweep")
        printlog("[State] Solve complete.")

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
        printlog("[State] Cleaning up current HFSS simulation...")
        try:
            if oDesign:
                if report_name in oReportModule.GetAllReportNames():
                    oReportModule.DeleteReports([report_name])
                oDesign.DeleteFullVariation("All", False)

            if oEditor:
                oEditor.Delete(
                    [
                        "NAME:Selections",
                        "Selections:=", HORN_OBJECT_NAME
                    ])
                printlog("[State] Successfully cleaned up {}".format(HORN_OBJECT_NAME))
        except Exception as cleanup_e:
            printlog("[ERROR] HFSS object cleanup: {}".format(cleanup_e))


# --- Main Loop ---
printlog("[State] Entering main loop...")

while True:
    if os.path.exists(DONE_FLAG_FILE):
        printlog("[State] Done flag detected. Exiting subprocess loop.")
        break

    if _all_model_files_ready(MODEL_FILE):
        printlog("[State] Detected all horn model files. Starting simulation run.")
        time.sleep(0.2)
        runSimulation()

        if os.path.exists(DONE_FLAG_FILE):
            printlog("[State] Done flag detected after simulation run.")
            break

    time.sleep(1)

printlog("--- All Completed ---")
