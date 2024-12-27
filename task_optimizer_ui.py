from task_optimizer import TaskOptimizer 
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import datetime
from typing import Optional
import json

class TaskOptimizerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Task Optimizer")
        self.root.geometry("1200x800")
        
        # Initialize the task optimizer
        self.optimizer = TaskOptimizer()
        
        # Video capture
        self.cap = None
        self.video_queue = queue.Queue()
        self.is_webcam_active = False
        
        # Current employee
        self.current_employee_id = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main UI components"""
        # Create main containers
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.right_frame = ttk.Frame(self.root, padding="10")
        
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_employee_section()
        self.setup_video_section()
        self.setup_mood_section()
        self.setup_task_section()
        self.setup_team_analytics_section()
        
    def setup_employee_section(self):
        """Set up employee management section"""
        employee_frame = ttk.LabelFrame(self.left_frame, text="Employee Management", padding="10")
        employee_frame.pack(fill=tk.X, pady=5)
        
        # Employee ID entry
        ttk.Label(employee_frame, text="Employee ID:").pack(side=tk.LEFT)
        self.employee_id_var = tk.StringVar()
        self.employee_id_entry = ttk.Entry(employee_frame, textvariable=self.employee_id_var)
        self.employee_id_entry.pack(side=tk.LEFT, padx=5)
        
        # Register button
        self.register_btn = ttk.Button(
            employee_frame,
            text="Register/Select Employee",
            command=self.register_employee
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
    def setup_video_section(self):
        """Set up video feed section"""
        video_frame = ttk.LabelFrame(self.left_frame, text="Video Feed", padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480)
        self.video_canvas.pack()
        
        # Control buttons
        btn_frame = ttk.Frame(video_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Camera",
            command=self.toggle_camera
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(
            btn_frame,
            text="Capture State",
            command=self.capture_state,
            state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
    def setup_mood_section(self):
        """Set up mood display section"""
        mood_frame = ttk.LabelFrame(self.right_frame, text="Current Mood", padding="10")
        mood_frame.pack(fill=tk.X, pady=5)
        
        self.mood_label = ttk.Label(mood_frame, text="No mood detected")
        self.mood_label.pack()
        
    def setup_task_section(self):
        """Set up task recommendation section"""
        task_frame = ttk.LabelFrame(self.right_frame, text="Task Recommendations", padding="10")
        task_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Task list
        self.task_list = tk.Listbox(task_frame, height=6)
        self.task_list.pack(fill=tk.BOTH, expand=True)
        
    def setup_team_analytics_section(self):
        """Set up team analytics section"""
        analytics_frame = ttk.LabelFrame(self.right_frame, text="Team Analytics", padding="10")
        analytics_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.analytics_text = tk.Text(analytics_frame, height=10, wrap=tk.WORD)
        self.analytics_text.pack(fill=tk.BOTH, expand=True)
        
    def register_employee(self):
        """Register or select an employee"""
        employee_id = self.employee_id_var.get().strip()
        if not employee_id:
            messagebox.showerror("Error", "Please enter an employee ID")
            return
            
        self.current_employee_id = employee_id
        self.optimizer.register_employee(employee_id)
        self.capture_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"Employee {employee_id} registered/selected")
        
    def toggle_camera(self):
        """Toggle webcam on/off"""
        if not self.is_webcam_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
                
            self.is_webcam_active = True
            self.start_btn.config(text="Stop Camera")
            threading.Thread(target=self.update_video, daemon=True).start()
        else:
            self.is_webcam_active = False
            self.start_btn.config(text="Start Camera")
            if self.cap:
                self.cap.release()
                
    def update_video(self):
        """Update video feed"""
        while self.is_webcam_active:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                # Resize to fit canvas
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas
                self.video_queue.put(photo)
                self.root.after(1, self.update_canvas)
                
    def update_canvas(self):
        """Update the video canvas with the latest frame"""
        try:
            photo = self.video_queue.get_nowait()
            self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.video_canvas.image = photo
        except queue.Empty:
            pass
            
    def capture_state(self):
        """Capture current state and update recommendations"""
        if not self.current_employee_id:
            messagebox.showerror("Error", "Please register/select an employee first")
            return
            
        if not self.cap or not self.is_webcam_active:
            messagebox.showerror("Error", "Please start the camera first")
            return
            
        # Capture current frame
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not capture frame")
            return
            
        # Process employee state
        result = self.optimizer.process_employee_state(
            self.current_employee_id,
            frame
        )
        
        # Update UI
        self.update_results(result)
        
        # Update team analytics
        self.update_team_analytics()
        
    def update_results(self, result: dict):
        """Update UI with new results"""
        # Update mood
        self.mood_label.config(
            text=f"Current Mood: {result['current_mood']}\n"
                f"Stress Level: {result['mood_analysis']['stress_level']:.2f}"
        )
        
        # Update task recommendations
        self.task_list.delete(0, tk.END)
        for task in result['task_recommendations']:
            self.task_list.insert(tk.END, task)
            
        # Show alert if required
        if result['alert_required']:
            messagebox.showwarning(
                "Stress Alert",
                "Employee stress levels are high. Please check in with them."
            )
            
    def update_team_analytics(self):
        """Update team analytics display"""
        team_result = self.optimizer.get_team_analytics([self.current_employee_id])
        
        self.analytics_text.delete(1.0, tk.END)
        self.analytics_text.insert(tk.END, json.dumps(team_result, indent=2))

def main():
    root = tk.Tk()
    app = TaskOptimizerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()