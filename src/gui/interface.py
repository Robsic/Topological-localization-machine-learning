import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

from src.gui.utils_gui import parse_ini_file
from src.gui.map_viewer import generate_static_map_image
from main import *


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Topological Localization")
        self.root.geometry("1400x900")

        self.nodes = []
        self.mode = None
        self.stop_requested = False

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.button_style = {"font": ("Arial", 12), "width": 20, "height": 2}

        self.show_start_screen()

    def show_start_screen(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        tk.Button(
            self.main_frame,
            text="Start",
            command=self.select_ini_file,
            **self.button_style
        ).pack(expand=True)

    def select_ini_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("INI files", "*.ini")])
        if not filepath:
            return

        try:
            self.nodes = parse_ini_file(filepath)
            self.build_main_screen()
        except Exception as e:
            messagebox.showerror("Failed to load .ini file", str(e))
            self.show_start_screen()

    def build_main_screen(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        sidebar = tk.Frame(self.main_frame, width=150, bg="lightgray")
        sidebar.pack(side="right", fill="y")

        tk.Button(
            sidebar,
            text="Navigate",
            command=self.show_navigation_options,
            **self.button_style
        ).pack(pady=10, padx=10)

        tk.Button(
            sidebar,
            text="Start Capture",
            command=self.show_capture_options,
            **self.button_style
        ).pack(pady=10, padx=10)

        tk.Button(
            sidebar,
            text="Select New Map",
            command=self.show_start_screen,
            **self.button_style
        ).pack(pady=10, padx=10)

        try:
            map_path = generate_static_map_image(self.nodes)
            map_img = tk.PhotoImage(file=map_path)

            map_label = tk.Label(self.main_frame, image=map_img)
            map_label.image = map_img
            map_label.pack(side="left", fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Failed to generate map", str(e))
            self.show_start_screen()

    def not_implemented(self, msg="Feature under development."):
        messagebox.showinfo("Info", msg)

    def show_navigation_options(self):
        nav_window = tk.Toplevel(self.root)
        nav_window.title("Navigation Mode")
        nav_window.geometry("500x300")

        def navigation_under_development():
            nav_window.destroy()
            self.not_implemented()

        def launch_free_route():
            nav_window.destroy()
            self.start_free_route()

        tk.Button(
            nav_window,
            text="Navigation",
            command=navigation_under_development,
            **self.button_style
        ).pack(pady=10)

        tk.Button(
            nav_window,
            text="Free Route",
            command=launch_free_route,
            **self.button_style
        ).pack(pady=10)

    def show_capture_options(self):
        options_window = tk.Toplevel(self.root)
        options_window.title("Capture Mode")
        options_window.geometry("500x300")

        def launch_capture_and_close(mode):
            options_window.destroy()
            self.start_capture(mode)

        tk.Button(
            options_window,
            text="All Points",
            command=lambda: launch_capture_and_close("all"),
            **self.button_style
        ).pack(pady=10)

        tk.Button(
            options_window,
            text="Only Nodes",
            command=lambda: launch_capture_and_close("nodes_only"),
            **self.button_style
        ).pack(pady=10)

    def start_capture(self, mode):
        self.mode = mode
        self.stop_requested = False

        def should_stop():
            return self.stop_requested

        def on_stop():
            self.root.after(0, self.build_main_screen)

        Thread(
            target=lambda: start_capture(self.nodes, mode, should_stop, on_stop),
            daemon=True
        ).start()

        self.capture_window = tk.Toplevel(self.root)
        self.capture_window.title("Capture Running")
        self.capture_window.geometry("400x200")
        self.capture_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

        tk.Label(
            self.capture_window,
            text="Capture in progress...\nClose this window or click Stop to interrupt.",
            font=("Arial", 12)
        ).pack(pady=30)

        tk.Button(
            self.capture_window,
            text="Stop Capture",
            command=self.stop_capture,
            **self.button_style
        ).pack()

    def start_free_route(self):
        self.stop_requested = False

        def should_stop():
            return self.stop_requested

        def on_stop():
            self.root.after(0, self.build_main_screen)

        Thread(
            target=lambda: start_free_route(nodes=self.nodes, should_stop=should_stop, on_stop_callback=on_stop),
            daemon=True
        ).start()

        self.capture_window = tk.Toplevel(self.root)
        self.capture_window.title("Free Route Running")
        self.capture_window.geometry("400x200")
        self.capture_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

        tk.Label(
            self.capture_window,
            text="Free route logging in progress...\nClose this window or click Stop to interrupt.",
            font=("Arial", 12)
        ).pack(pady=30)

        tk.Button(
            self.capture_window,
            text="Stop",
            command=self.stop_capture,
            **self.button_style
        ).pack()

    def stop_capture(self):
        self.stop_requested = True
        if hasattr(self, "capture_window") and self.capture_window.winfo_exists():
            self.capture_window.destroy()
        self.root.after(100, self.build_main_screen)



def run_gui():
    """GUI entry point."""
    root = tk.Tk()
    app = App(root)
    root.mainloop()
