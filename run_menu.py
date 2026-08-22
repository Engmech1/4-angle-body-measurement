"""
Exercise App - Interactive Master Menu & Application Launcher.

Provides a clean, crash-proof interactive terminal menu (Thai & English)
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
        print(f"\n[ERROR] Failed to run {script_name}: {e}")
    print("\n" + "-" * 75)
    input("กดปุ่ม Enter เพื่อกลับสู่เมนูหลัก (Press Enter to return to menu)...")


def run_pytest():
    clear_screen()
    cmd = [sys.executable, "-m", "pytest", "-v"]
    print("=" * 75)
    print(f"  RUNNING PYTEST TEST SUITE: {' '.join(cmd)}")
    print("=" * 75 + "\n")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"\n[ERROR] Test suite execution failed: {e}")
    print("\n" + "-" * 75)
    input("กดปุ่ม Enter เพื่อกลับสู่เมนูหลัก (Press Enter to return to menu)...")


def show_readme():
    clear_screen()
    if os.path.exists("README.md"):
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            print(content[:3000])  # Show preview
            print("\n... [ดูเนื้อหาฉบับเต็มได้ที่ไฟล์ README.md] ...\n")
        except Exception as e:
            print(f"Could not open README.md: {e}")
    else:
        print("README.md not found.")
    print("-" * 75)
    input("กดปุ่ม Enter เพื่อกลับสู่เมนูหลัก (Press Enter to return to menu)...")


def main():
    while True:
        clear_screen()
        print("=" * 75)
        print("       EXERCISE APP: 4-ANGLE GUIDED CAPTURE + ML MEASUREMENT SYSTEM")
        print("                  High-Precision Computer Vision & ML Engine")
        print("=" * 75)
        print(f"  Python Environment: {sys.executable}")
        print("=" * 75)
        print("\n  กรุณาเลือกเมนูที่ต้องการรัน (Select an option):\n")
        print("   [1] Multi-View Calibration Dashboard (calibration_dashboard.py)")
        print("       - หน้าจอวิเคราะห์ 4 ช่อง Real-time (ArUco, Skeleton, Oscilloscope, Cross-Section)")
        print("       - รองรับกล้องคอมพิวเตอร์ และกล้องมือถือ (DroidCam / IP Webcam)\n")
        print("   [2] Live Computer Webcam Capture + ML Engine (live_camera_capture.py)")
        print("       - ถ่ายวัดสรีระ 4 มุมอัตโนมัติ พร้อมระบบ Online ML Active Learning\n")
        print("   [3] Interactive Guided Capture Demo (demo_capture.py)")
        print("       - จำลองขั้นตอนการวัดสรีระ 4 มุม (0, 90, 180, 270 องศา) ในหน่วยความจำ\n")
        print("   [4] Adversarial QA Benchmark (run_simulation.py)")
        print("       - ทดสอบความแม่นยำ Error < 0.5 cm บน 6 สรีระมนุษย์ พร้อม Sway และ Noise\n")
        print("   [5] Generate Visual Diagnostic Report & Plots (visualize_pipeline.py)")
        print("       - สร้างรูปภาพกราฟวิเคราะห์ 4 มิติ (body_measurement_visual_report.png)\n")
        print("   [6] Train / Retrain ML Continual Learning Model (train_ml_model.py)")
        print("       - ฝึกโมเดล ML เรียนรู้ข้อผิดพลาดทางสรีระและชดเชยมิติอัตโนมัติ\n")
        print("   [7] Run Pytest Unit & Integration Tests (pytest -v)")
        print("       - ตรวจสอบความถูกต้องของทุกโมดูล (Sub-Pixel, Scaling, MAD, Spline, ML, SMPL)\n")
        print("   [8] View README.md Documentation\n")
        print("   [0] Exit (ออกจากโปรแกรม)\n")
        print("=" * 75)

        try:
            choice = input("ป้อนหมายเลขเมนู (0-8): ").strip()
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
            print("\nขอบคุณที่ใช้งาน Exercise App Body Measurement Engine!\n")
            time.sleep(1.0)
            break
        else:
            print("\n[ERROR] หมายเลขเมนูไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง...")
            time.sleep(1.2)


if __name__ == "__main__":
    main()
