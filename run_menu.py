"""
Exercise App - Interactive Master Menu & Application Launcher.

Provides a robust, clean interactive terminal menu (100% English)
to launch all modules, dashboards, benchmarks, and tests.
"""

import os
import subprocess
import sys
import time

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def run_script(script_name: str, args: list = None):
    clear_screen()
    cmd = [sys.executable, script_name] + (args or [])
    print("=" * 75)
    print(f"  EXECUTING: {' '.join(cmd)}")
    print("=" * 75 + "\n")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"\n[ERROR] Failed to execute {script_name}: {e}")
    print("\n" + "-" * 75)
    input("Press Enter to return to the Main Menu...")


def run_pytest():
    clear_screen()
    cmd = [sys.executable, "-m", "pytest", "-v"]
    print("=" * 75)
    print(f"  RUNNING TEST SUITE: {' '.join(cmd)}")
    print("=" * 75 + "\n")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
    print("\n" + "-" * 75)
    input("Press Enter to return to the Main Menu...")


def show_readme():
    clear_screen()
    if os.path.exists("README.md"):
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            print(content[:3000])
            print("\n... [View full documentation in README.md file] ...\n")
        except Exception as e:
            print(f"Could not open README.md: {e}")
    else:
        print("README.md not found.")
    print("-" * 75)
    input("Press Enter to return to the Main Menu...")


def main():
    while True:
        clear_screen()
        print("=" * 75)
        print("       EXERCISE APP: 4-ANGLE GUIDED CAPTURE + ML MEASUREMENT SYSTEM")
        print("                  High-Precision Computer Vision & ML Engine")
        print("=" * 75)
        print(f"  Python Environment: {sys.executable}")
        print("=" * 75)
        print("\n  Please select an option to launch:\n")
        print("   [1] Multi-View Calibration Dashboard (calibration_dashboard.py)")
        print("       - 4-Quadrant Real-Time Diagnostic View (ArUco, Skeleton, DoG, Spline)")
        print("       - Supports PC Webcam & Smartphone IP Cameras (DroidCam / IP Webcam)\n")
        print("   [2] Live Computer Webcam Capture + ML Engine (live_camera_capture.py)")
        print("       - Guided 4-Angle Body Measurement with Online ML Active Learning\n")
        print("   [3] Interactive Guided Capture Demo (demo_capture.py)")
        print("       - In-Memory 4-Angle Guided Capture Simulation (0, 90, 180, 270 deg)\n")
        print("   [4] Adversarial QA Benchmark (run_simulation.py)")
        print("       - 6 Diverse Human Somatotypes Error Benchmark (< 0.5 cm Target)\n")
        print("   [5] Generate Visual Diagnostic Report & Plots (visualize_pipeline.py)")
        print("       - Generates 4-Panel Analysis Figure (body_measurement_visual_report.png)\n")
        print("   [6] Train / Retrain ML Continual Learning Model (train_ml_model.py)")
        print("       - Fits Non-Elliptical Morphology & Residual Bias Optimizer\n")
        print("   [7] Run Pytest Unit & Integration Test Suite (pytest -v)")
        print("       - Comprehensive Verification across all Modules (Sub-Pixel, MAD, ML, SMPL)\n")
        print("   [8] View README.md Documentation\n")
        print("   [0] Exit Application\n")
        print("=" * 75)

        try:
            choice = input("Enter choice (0-8): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if choice == "1":
            run_script("calibration_dashboard.py")
        elif choice == "2":
            run_script("live_camera_capture.py")
        elif choice == "3":
            run_script("demo_capture.py")
        elif choice == "4":
            run_script("run_simulation.py")
        elif choice == "5":
            run_script("visualize_pipeline.py")
        elif choice == "6":
            run_script("train_ml_model.py")
        elif choice == "7":
            run_pytest()
        elif choice == "8":
            show_readme()
        elif choice == "0":
            clear_screen()
            print("\nThank you for using the Exercise App Body Measurement Engine!\n")
            time.sleep(0.8)
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number between 0 and 8.")
            time.sleep(1.2)


if __name__ == "__main__":
    main()
