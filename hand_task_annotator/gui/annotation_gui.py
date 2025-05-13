"""
GUI for the hand task annotation system
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import time
import queue
import json

from hand_task_annotator.core.annotator import VideoAnnotator


class AnnotationGUI:
    """
    Provides a graphical user interface for the video annotation system.
    
    Features include:
    - File selection for input and output
    - Configuration of sampling rate
    - Debug mode toggle
    - Real-time progress display
    - Activity visualization
    - Start/stop controls
    """
    
    def __init__(self, master):
        self.master = master
        master.title("Hand Task Video Annotator")
        master.geometry("800x700")
        
        # Set up styles
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=6)
        self.style.configure("TFrame", padding=10)
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.sampling_rate = tk.IntVar(value=25)  # Default 25Hz
        self.debug_mode = tk.BooleanVar(value=False)
        self.real_time_output = tk.BooleanVar(value=True)  # Default to real-time output
        self.use_ocr = tk.BooleanVar(value=False)  # OCR timestamp extraction
        self.frame_delay = tk.IntVar(value=1)  # Speed control (ms)
        
        # For storing default paths
        self.config_file = os.path.expanduser("~/.hand_annotator_config.json")
        self.load_config()
        
        # Create main frame with padding
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        ttk.Button(file_frame, text="Browse...", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Output CSV:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Configure columm weights for file_frame
        file_frame.columnconfigure(1, weight=1)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # First row: Sampling rate
        ttk.Label(settings_frame, text="Sampling Rate (Hz):").grid(row=0, column=0, sticky=tk.W)
        sampling_spin = ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.sampling_rate, width=5)
        sampling_spin.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Debug mode checkbox
        ttk.Checkbutton(settings_frame, text="Debug Mode", variable=self.debug_mode).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Real-time output checkbox
        ttk.Checkbutton(settings_frame, text="Real-time Output", variable=self.real_time_output).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Second row: OCR and Visualization settings
        ttk.Checkbutton(settings_frame, text="OCR Timestamps", variable=self.use_ocr).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Visualization Speed:").grid(row=1, column=1, sticky=tk.W)
        ttk.Scale(settings_frame, from_=1, to=100, orient="horizontal", variable=self.frame_delay).grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=15)
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(pady=5)
        
        self.activity_label = ttk.Label(progress_frame, text="No activity", font=("TkDefaultFont", 14, "bold"))
        self.activity_label.pack(pady=5)
        
        # Status indicator (colored frame)
        self.status_indicator = tk.Frame(progress_frame, width=50, height=20, bg="gray")
        self.status_indicator.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Annotation", command=self.start_annotation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_annotation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Console", command=self.clear_console).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_application).pack(side=tk.RIGHT, padx=5)
        
        # Initialize variables
        self.annotator = None
        self.annotation_thread = None
        self.running = False
        self.msg_queue = queue.Queue()
        
        # Start message polling
        self.master.after(100, self.process_messages)
        
    def load_config(self):
        """Load saved configuration if it exists"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if 'last_video_dir' in config and os.path.exists(config['last_video_dir']):
                        self.last_video_dir = config['last_video_dir']
                    else:
                        self.last_video_dir = os.path.expanduser("~")
                        
                    if 'last_output_dir' in config and os.path.exists(config['last_output_dir']):
                        self.last_output_dir = config['last_output_dir']
                    else:
                        self.last_output_dir = os.path.expanduser("~")
            else:
                self.last_video_dir = os.path.expanduser("~")
                self.last_output_dir = os.path.expanduser("~")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self.last_video_dir = os.path.expanduser("~")
            self.last_output_dir = os.path.expanduser("~")
    
    def save_config(self):
        """Save configuration settings"""
        try:
            config = {
                'last_video_dir': self.last_video_dir,
                'last_output_dir': self.last_output_dir
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {str(e)}")
    
    def browse_video(self):
        """Open file dialog to select input video"""
        filepath = filedialog.askopenfilename(
            initialdir=self.last_video_dir,
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if filepath:
            self.video_path.set(filepath)
            self.last_video_dir = os.path.dirname(filepath)
            self.save_config()
            
            # Auto-generate output path with same name but .csv extension
            if not self.output_path.get():
                output = os.path.splitext(filepath)[0] + "_annotations.csv"
                self.output_path.set(output)
    
    def browse_output(self):
        """Open file dialog to select output CSV file"""
        filepath = filedialog.asksaveasfilename(
            initialdir=self.last_output_dir,
            title="Select Output CSV File",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.output_path.set(filepath)
            self.last_output_dir = os.path.dirname(filepath)
            self.save_config()
    
    def write_to_console(self, message, level="INFO"):
        """Write a message to the console with color coding"""
        self.console.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Color-code by level
        if level == "ERROR":
            tag = "error"
            color = "red"
        elif level == "WARNING":
            tag = "warning"
            color = "orange"
        elif level == "SUCCESS":
            tag = "success"
            color = "green"
        elif level == "DEBUG":
            tag = "debug"
            color = "blue"
        else:
            tag = "info"
            color = "black"
            
        # Create tag if it doesn't exist
        if tag not in self.console.tag_names():
            self.console.tag_configure(tag, foreground=color)
            
        # Insert text with tag
        self.console.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.console.insert(tk.END, f"[{level}] ", tag)
        self.console.insert(tk.END, message + "\n")
        
        # Scroll to bottom
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
    
    def clear_console(self):
        """Clear the console output"""
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)
    
    def update_progress(self, value, activity, timestamp):
        """Update progress bar and activity display"""
        self.progress["value"] = value
        
        # Format timestamp
        minutes, seconds = divmod(int(timestamp), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        self.progress_label.config(text=f"Progress: {value:.1f}% - Time: {time_str}")
        self.activity_label.config(text=f"Activity: {activity}")
        
        # Update status indicator color
        if activity == "Off Task":
            self.status_indicator.config(bg="red")
        elif "Taking" in activity:
            self.status_indicator.config(bg="orange")
        else:
            self.status_indicator.config(bg="green")
    
    def start_annotation(self):
        """Start the annotation process in a separate thread"""
        # Validate inputs
        if not self.video_path.get():
            self.write_to_console("No video file specified", "ERROR")
            return
            
        if not self.output_path.get():
            self.write_to_console("No output file specified", "ERROR")
            return
        
        # Disable start button, enable stop button
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        
        # Create annotator
        self.annotator = VideoAnnotator(
            video_path=self.video_path.get(),
            sampling_rate=self.sampling_rate.get(),
            output_path=self.output_path.get(),
            real_time_output=self.real_time_output.get(),
            debug_mode=self.debug_mode.get(),
            log_file=None  # No file logging in GUI mode
        )
        
        # Set visualization speed
        self.annotator.frame_delay = self.frame_delay.get()
        
        # Override annotator's log function
        self.annotator.log = self.log_to_queue
        
        # Start in a separate thread
        self.annotation_thread = threading.Thread(
            target=self.run_annotation,
            args=(self.video_path.get(), self.output_path.get())
        )
        self.annotation_thread.daemon = True
        self.annotation_thread.start()
        
        self.write_to_console("Annotation started", "SUCCESS")
    
    def log_to_queue(self, message, level="INFO"):
        """Add log message to queue to be processed in the main thread"""
        self.msg_queue.put((message, level))
    
    def process_messages(self):
        """Process any pending messages in the queue"""
        try:
            while True:
                message, level = self.msg_queue.get_nowait()
                self.write_to_console(message, level)
        except queue.Empty:
            pass
        
        # Schedule to run again
        self.master.after(100, self.process_messages)
    
    def run_annotation(self, video_path, output_path):
        """Run the annotation process"""
        try:
            result = self.annotator.process_video(
                output_path=output_path,
                gui_callback=self.callback_from_thread
            )
            
            if result:
                self.msg_queue.put(("Annotation completed successfully", "SUCCESS"))
            else:
                self.msg_queue.put(("Annotation failed", "ERROR"))
        except Exception as e:
            import traceback
            self.msg_queue.put((f"Error: {str(e)}\n{traceback.format_exc()}", "ERROR"))
        finally:
            # Reset UI state
            self.running = False
            self.master.after(0, self.reset_buttons)
    
    def callback_from_thread(self, progress, activity, timestamp):
        """Callback from the worker thread to update UI in main thread"""
        self.master.after(0, lambda: self.update_progress(progress, activity, timestamp))
    
    def reset_buttons(self):
        """Reset button states after annotation completes"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def stop_annotation(self):
        """Stop the annotation process"""
        if self.annotator:
            # Signal the annotator to stop
            self.running = False
            
            # Wait for thread to finish
            if self.annotation_thread and self.annotation_thread.is_alive():
                self.write_to_console("Stopping annotation process...", "WARNING")
                self.annotation_thread.join(0.1)  # Brief timeout
            
            self.reset_buttons()
            self.write_to_console("Annotation stopped by user", "WARNING")
    
    def exit_application(self):
        """Clean up and exit"""
        self.stop_annotation()
        self.save_config()
        self.master.destroy()


def run_gui():
    """Start the GUI application"""
    root = tk.Tk()
    app = AnnotationGUI(root)
    root.mainloop() 