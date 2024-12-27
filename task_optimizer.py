import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
import speech_recognition as sr
from textblob import TextBlob
import hashlib
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmployeeData:
    def __init__(self, employee_id: str):
        self.employee_id = self._hash_id(employee_id)
        self.mood_history = []
        self.task_history = []
        
    @staticmethod
    def _hash_id(employee_id: str) -> str:
        """Hash employee ID for privacy"""
        return hashlib.sha256(employee_id.encode()).hexdigest()
    
    def add_mood_entry(self, mood: str, timestamp: datetime.datetime):
        """Add a new mood entry with timestamp"""
        self.mood_history.append({
            'timestamp': timestamp,
            'mood': mood
        })
        
    def add_task_entry(self, task: str, completion_status: bool, timestamp: datetime.datetime):
        """Add a task entry with completion status"""
        self.task_history.append({
            'timestamp': timestamp,
            'task': task,
            'completed': completion_status
        })

class EmotionDetector:
    def __init__(self):
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.speech_recognizer = sr.Recognizer()
        
    def detect_facial_emotion(self, frame) -> str:
        """
        Detect emotion from facial expressions using simple heuristics
        Returns: predicted emotion (happy, sad, neutral, stressed)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # For demonstration, using a simple random distribution
                # In a real implementation, you'd want to use a proper emotion detection model
                emotions = ['happy', 'neutral', 'sad', 'stressed']
                weights = [0.4, 0.3, 0.2, 0.1]  # Example probability distribution
                return np.random.choice(emotions, p=weights)
            
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error in facial emotion detection: {str(e)}")
            return 'neutral'
    
    def detect_speech_emotion(self, audio_file: str) -> str:
        """
        Detect emotion from speech
        Returns: predicted emotion based on speech analysis
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.speech_recognizer.record(source)
                text = self.speech_recognizer.recognize_google(audio)
                
                # Analyze sentiment using TextBlob
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.3:
                    return 'happy'
                elif polarity < -0.3:
                    return 'sad'
                else:
                    return 'neutral'
                
        except Exception as e:
            logger.error(f"Error in speech emotion detection: {str(e)}")
            return 'neutral'

# [Rest of the classes remain the same as in the previous version]
class TaskRecommender:
    def __init__(self):
        self.task_categories = {
            'happy': ['complex problem solving', 'creative tasks', 'team collaboration'],
            'neutral': ['routine tasks', 'documentation', 'planning'],
            'sad': ['simple tasks', 'organizing', 'skill development'],
            'stressed': ['breaks', 'low-pressure tasks', 'administrative work']
        }
        
    def get_task_recommendation(self, mood: str) -> List[str]:
        """Get task recommendations based on current mood"""
        return self.task_categories.get(mood, self.task_categories['neutral'])
    
    def analyze_task_history(self, task_history: List[Dict]) -> Dict:
        """Analyze task completion patterns"""
        df = pd.DataFrame(task_history)
        completion_rate = df['completed'].mean()
        return {
            'completion_rate': completion_rate,
            'total_tasks': len(task_history),
            'completed_tasks': df['completed'].sum()
        }

class MoodAnalytics:
    def __init__(self):
        self.stress_threshold = 3  # Number of consecutive stressed/sad days
        
    def analyze_mood_trends(self, mood_history: List[Dict]) -> Dict:
        """Analyze mood patterns and identify concerning trends"""
        df = pd.DataFrame(mood_history)
        
        if len(df) < 1:
            return {'status': 'insufficient_data'}
            
        recent_moods = df.tail(self.stress_threshold)
        stress_count = sum(1 for mood in recent_moods['mood'] if mood in ['stressed', 'sad'])
        
        return {
            'status': 'alert' if stress_count >= self.stress_threshold else 'normal',
            'stress_level': stress_count / self.stress_threshold,
            'dominant_mood': df['mood'].mode().iloc[0],
            'mood_variability': len(df['mood'].unique()) / len(df)
        }

class TaskOptimizer:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.task_recommender = TaskRecommender()
        self.mood_analytics = MoodAnalytics()
        self.employees = {}
        
    def register_employee(self, employee_id: str):
        """Register a new employee"""
        if employee_id not in self.employees:
            self.employees[employee_id] = EmployeeData(employee_id)
            logger.info(f"Registered new employee with hashed ID: {self.employees[employee_id].employee_id}")
    
    def process_employee_state(self, 
                             employee_id: str,
                             video_frame,
                             audio_file: Optional[str] = None) -> Dict:
        """Process employee state and provide recommendations"""
        if employee_id not in self.employees:
            self.register_employee(employee_id)
            
        # Detect emotions
        facial_emotion = self.emotion_detector.detect_facial_emotion(video_frame)
        speech_emotion = self.emotion_detector.detect_speech_emotion(audio_file) if audio_file else 'neutral'
        
        # Combine emotions (simple average - could be more sophisticated)
        final_mood = facial_emotion if facial_emotion == speech_emotion else 'neutral'
        
        # Update employee records
        self.employees[employee_id].add_mood_entry(final_mood, datetime.datetime.now())
        
        # Get recommendations and analytics
        recommendations = self.task_recommender.get_task_recommendation(final_mood)
        mood_analysis = self.mood_analytics.analyze_mood_trends(
            self.employees[employee_id].mood_history
        )
        
        return {
            'current_mood': final_mood,
            'task_recommendations': recommendations,
            'mood_analysis': mood_analysis,
            'alert_required': mood_analysis['status'] == 'alert'
        }
    
    def get_team_analytics(self, team_ids: List[str]) -> Dict:
        """Get aggregated team mood analytics"""
        team_moods = []
        for emp_id in team_ids:
            if emp_id in self.employees:
                emp_data = self.employees[emp_id]
                if emp_data.mood_history:
                    team_moods.append(emp_data.mood_history[-1]['mood'])
        
        if not team_moods:
            return {'status': 'no_data'}
        
        return {
            'team_mood_distribution': {
                mood: team_moods.count(mood) / len(team_moods)
                for mood in set(team_moods)
            },
            'team_stress_level': sum(1 for mood in team_moods if mood == 'stressed') / len(team_moods)
        }

# Example usage
def main():
    # Initialize the optimizer
    optimizer = TaskOptimizer()
    
    # Mock video frame (in practice, this would come from a camera)
    mock_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Process an employee
    result = optimizer.process_employee_state(
        employee_id="EMP001",
        video_frame=mock_frame
    )
    
    # Print results
    print("Employee Analysis Results:")
    print(json.dumps(result, indent=2))
    
    # Get team analytics
    team_result = optimizer.get_team_analytics(["EMP001", "EMP002"])
    print("\nTeam Analysis Results:")
    print(json.dumps(team_result, indent=2))

if __name__ == "__main__":
    main()