import os
import sys
import json
import math
import time
import copy
import uuid
import random
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / abs(b) * 100.0

@dataclass
class UserProfile:
    user_id: str
    age: int
    gender: str
    height: float
    weight: float
    fitness_level: str
    goals: List[str]
    medical_conditions: List[str]
    preferred_workout_types: List[str]
    available_equipment: List[str]
    time_availability: int
    experience_years: float

    def bmi(self) -> float:
        h_m = self.height / 100.0
        return self.weight / (h_m * h_m)

    def bmr(self) -> float:
        if self.gender.lower() == "male":
            return 88.362 + 13.397 * self.weight + 4.799 * self.height - 5.677 * self.age
        return 447.593 + 9.247 * self.weight + 3.098 * self.height - 4.330 * self.age


@dataclass
class Exercise:
    exercise_id: str
    name: str
    category: str
    muscle_groups: List[str]
    equipment_needed: List[str]
    difficulty_level: str
    calories_per_minute: float
    instructions: str
    modifications: Dict[str, str]


@dataclass
class WorkoutSession:
    session_id: str
    user_id: str
    date: datetime.date
    exercises: List[Dict]
    duration: int
    calories_burned: int
    difficulty_rating: int
    completion_status: str
    user_feedback: int
    heart_rate_avg: Optional[int] = None
    notes: str = ""

class FitnessDatabase:
    def __init__(self):
        self.exercises: Dict[str, Exercise] = self._seed_exercises()
        self.users: Dict[str, UserProfile] = {}
        self.workouts: Dict[str, List[WorkoutSession]] = {}

    def add_user(self, profile: UserProfile):
        self.users[profile.user_id] = profile
        self.workouts.setdefault(profile.user_id, [])

    def log_session(self, session: WorkoutSession):
        self.workouts.setdefault(session.user_id, []).append(session)

    def history(self, user_id: str, days: int = 90) -> List[WorkoutSession]:
        sessions = self.workouts.get(user_id, [])
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        return [s for s in sessions if s.date >= cutoff]

    def _seed_exercises(self) -> Dict[str, Exercise]:
        db = {}
        def add(eid, name, cat, muscles, equip, cpm, instr, mod, lvl="intermediate"):
            db[eid] = Exercise(
                exercise_id=eid,
                name=name,
                category=cat,
                muscle_groups=muscles,
                equipment_needed=equip,
                difficulty_level=lvl,
                calories_per_minute=cpm,
                instructions=instr,
                modifications=mod
            )

        # Cardio
        add("jumping_jacks", "Jumping Jacks", "cardio", ["full_body"], [],
            8.0, "Stand tall, jump feet out while raising arms; return and repeat.",
            {"beginner": "Go slower, 20s on 20s off.",
             "intermediate": "Steady pace, 45s on 15s off.",
             "advanced": "Max intensity, 60s on 10s off."})
        add("burpees", "Burpees", "cardio", ["full_body", "core"], [],
            12.0, "Squat, kick back to plank, push-up, return and jump explosively.",
            {"beginner": "Remove push-up, step back instead of jump.",
             "intermediate": "Standard burpee, steady cadence.",
             "advanced": "Add tuck jump and push-up on each rep."})
        add("mountain_climbers", "Mountain Climbers", "cardio", ["core", "legs"], [],
            10.0, "High plank; drive knees toward chest alternating quickly.",
            {"beginner": "Slow tempo, 20s on 20s off.",
             "intermediate": "45s on 15s off.",
             "advanced": "60s on 10s off, maintain hip stability."})
        add("high_knees", "High Knees", "cardio", ["legs", "core"], [],
            9.0, "Run in place lifting knees to hip height; drive arms.",
            {"beginner": "Low impact march.",
             "intermediate": "Controlled run at steady pace.",
             "advanced": "Sprint intervals, minimal ground contact."})
        add("running", "Running", "cardio", ["legs", "cardiovascular"], [],
            11.0, "Comfortable pace; keep relaxed shoulders and midfoot strike.",
            {"beginner": "Jog/walk intervals.",
             "intermediate": "Steady state 20-30 min.",
             "advanced": "Tempo or intervals, monitor pacing."})
        add("cycling", "Cycling", "cardio", ["legs", "cardiovascular"], ["bicycle"],
            8.0, "Maintain cadence; slight bend in knee at bottom of stroke.",
            {"beginner": "Flat route, light gear.",
             "intermediate": "Rolling terrain or cadence targets.",
             "advanced": "Intervals or hill repeats."})
        add("jump_rope", "Jump Rope", "cardio", ["legs", "arms", "cardiovascular"], ["jump_rope"],
            13.0, "Small hops, elbows close, wrists drive the rope.",
            {"beginner": "Single-unders with rest.",
             "intermediate": "Longer sets, pacing.",
             "advanced": "Double-unders or speed rounds."})
        add("shadow_boxing", "Shadow Boxing", "cardio", ["arms", "core", "cardiovascular"], [],
            10.0, "Jab-cross-hook combos; light footwork; guard up.",
            {"beginner": "Short rounds, focus on form.",
             "intermediate": "3 x 2-min rounds.",
             "advanced": "5 x 3-min rounds with slips/rolls."})

        # Strength
        add("push_ups", "Push-ups", "strength", ["chest", "arms", "core"], [],
            6.0, "Straight line from head to heels; chest near floor; press up.",
            {"beginner": "Knee or incline push-ups.",
             "intermediate": "Full ROM, 2-3 sets of 10-12.",
             "advanced": "Elevate feet or tempo reps."})
        add("squats", "Squats", "strength", ["legs", "glutes"], [],
            7.0, "Sit hips back; knees track over toes; chest tall.",
            {"beginner": "Box or assisted squats.",
             "intermediate": "Bodyweight, 3 x 12.",
             "advanced": "Add load or tempo."})
        add("lunges", "Lunges", "strength", ["legs", "glutes"], [],
            6.5, "Long stride; front knee over midfoot; lower with control.",
            {"beginner": "Static split squat holding support.",
             "intermediate": "Alternating lunges 3 x 10/leg.",
             "advanced": "Walking lunges with load."})
        add("plank", "Plank", "strength", ["core", "shoulders"], [],
            4.0, "Elbows under shoulders; brace core; neutral spine.",
            {"beginner": "Knees down holds.",
             "intermediate": "3 x 30-45s.",
             "advanced": "RKC plank or weighted."})
        add("deadlifts", "Deadlifts", "strength", ["back", "legs", "glutes"], ["dumbbells"],
            8.0, "Hinge at hips; neutral spine; drive through heels.",
            {"beginner": "Hip hinge drills with dowel.",
             "intermediate": "DB RDL 3 x 10-12.",
             "advanced": "Heavier loads, low reps."})
        add("bench_press", "Bench Press", "strength", ["chest", "arms"], ["barbell", "bench"],
            7.0, "Shoulder blades set; bar path mid-chest; full lockout.",
            {"beginner": "DB floor press.",
             "intermediate": "3 x 8-10 moderate weight.",
             "advanced": "Strength sets 4-6 reps."})
        add("pull_ups", "Pull-ups", "strength", ["back", "arms"], ["pull_up_bar"],
            9.0, "Full hang; pull chin over bar; control down.",
            {"beginner": "Assisted or negatives.",
             "intermediate": "3 x max reps.",
             "advanced": "Weighted sets."})
        add("dips", "Dips", "strength", ["arms", "chest"], ["dip_bars"],
            8.0, "Elbows back; shoulder depressed; full lockout.",
            {"beginner": "Bench dips with knees bent.",
             "intermediate": "Parallel bar dips 3 x 8-10.",
             "advanced": "Weighted dips."})
        add("shoulder_press", "Shoulder Press", "strength", ["shoulders", "arms"], ["dumbbells"],
            6.0, "Brace core; press overhead in straight path.",
            {"beginner": "Seated light DBs.",
             "intermediate": "3 x 10-12.",
             "advanced": "Standing heavy sets."})
        add("rows", "Bent-over Rows", "strength", ["back", "arms"], ["dumbbells"],
            7.0, "Hinge torso; pull elbows to hips; squeeze scapulae.",
            {"beginner": "Chest-supported rows.",
             "intermediate": "3 x 10-12.",
             "advanced": "Heavier loads."})

        # Flexibility
        add("yoga_flow", "Basic Yoga Flow", "flexibility", ["full_body"], ["yoga_mat"],
            3.0, "Cycle through gentle poses with breathing.",
            {"beginner": "Hold 20-30s.",
             "intermediate": "Vinyasa 3-5 breaths each.",
             "advanced": "Longer holds, deeper poses."}, lvl="beginner")
        add("stretching", "Full Body Stretch", "flexibility", ["full_body"], [],
            2.5, "Head-to-toe static stretch series.",
            {"beginner": "20s per stretch.",
             "intermediate": "30s per stretch.",
             "advanced": "45-60s per stretch."}, lvl="beginner")
        add("cat_cow", "Cat-Cow", "flexibility", ["back", "core"], [],
            2.0, "Alternate spinal flexion/extension with breath.",
            {"beginner": "Slow gentle cycles.",
             "intermediate": "Add thoracic mobility.",
             "advanced": "Longer breathing focus."}, lvl="beginner")
        add("child_pose", "Child's Pose", "flexibility", ["back", "hips"], [],
            1.5, "Hips to heels; reach forward; relax breaths.",
            {"beginner": "Short holds.",
             "intermediate": "Longer restorative holds.",
             "advanced": "Side reach variations."}, lvl="beginner")
        return db

class MLRecommendationEngine:
    def __init__(self, db: FitnessDatabase):
        self.db = db

    def _fitness_score(self, u: UserProfile) -> float:
        bmi = u.bmi()
        if 18.5 <= bmi <= 24.9:
            bmi_score = 100.0
        elif bmi < 18.5:
            bmi_score = _clamp(100.0 - (18.5 - bmi) * 10.0, 0.0, 100.0)
        else:
            bmi_score = _clamp(100.0 - (bmi - 24.9) * 5.0, 0.0, 100.0)
        exp_score = _clamp(u.experience_years * 20.0, 0.0, 100.0)
        lvl_map = {"beginner": 30.0, "intermediate": 65.0, "advanced": 90.0}
        lvl_score = lvl_map.get(u.fitness_level.lower(), 30.0)
        if 25 <= u.age <= 35:
            age_factor = 1.0
        elif u.age < 25:
            age_factor = 0.8 + (u.age - 18) * 0.03
        else:
            age_factor = max(0.6, 1.0 - (u.age - 35) * 0.01)
        return (bmi_score * 0.3 + exp_score * 0.3 + lvl_score * 0.4) * age_factor

    def _patterns(self, user_id: str) -> Dict:
        hist = self.db.history(user_id, days=90)
        if not hist:
            return {"preferred_duration": 30, "consistency": 0.0,
                    "exercise_satisfaction": {}, "best_time": "morning",
                    "total_workouts": 0, "avg_feedback": 0.0}
        avg_dur = np.mean([s.duration for s in hist])
        sat = {}
        for s in hist:
            for ex in s.exercises:
                n = ex.get("name", "unknown")
                sat.setdefault(n, []).append(s.user_feedback)
        distinct = len(set(s.date for s in hist))
        consistency = distinct / 90.0 * 100.0
        avg_fb = float(np.mean([s.user_feedback for s in hist]))
        return {
            "preferred_duration": int(avg_dur),
            "consistency": _clamp(consistency, 0, 100),
            "exercise_satisfaction": sat,
            "best_time": "morning",
            "total_workouts": len(hist),
            "avg_feedback": avg_fb
        }

    def _compatibility(self, ex: Exercise, u: UserProfile, pat: Dict) -> float:
        score = 50.0
        # equipment
        if all(eq in u.available_equipment or eq == "" for eq in ex.equipment_needed):
            score += 20.0
        else:
            score -= 30.0
        # level
        lm = {"beginner": 1, "intermediate": 2, "advanced": 3}
        ul = lm.get(u.fitness_level.lower(), 1)
        xl = lm.get(ex.difficulty_level.lower(), 2)
        score += (3 - abs(ul - xl)) * 10.0
        # goals
        gm = {
            "weight_loss": ["cardio"],
            "muscle_gain": ["strength"],
            "endurance": ["cardio", "strength"],
            "strength": ["strength"],
            "flexibility": ["flexibility"],
            "general_fitness": ["cardio", "strength", "flexibility"]
        }
        for g in u.goals:
            if g in gm and ex.category in gm[g]:
                score += 15.0
        # medical
        if "back_pain" in u.medical_conditions and ex.exercise_id in ["deadlifts", "rows"]:
            score -= 25.0
        if "knee_problems" in u.medical_conditions and ex.exercise_id in ["squats", "lunges", "running"]:
            score -= 25.0
        if "heart_condition" in u.medical_conditions and ex.category == "cardio":
            score -= 20.0
        # satisfaction
        sat = pat.get("exercise_satisfaction", {})
        if ex.name in sat:
            avg = float(np.mean(sat[ex.name]))
            score += (avg - 3.0) * 10.0
        return _clamp(score, 0.0, 100.0)

    def recommend(self, user_id: str, target_duration: Optional[int] = None) -> Dict:
        if user_id not in self.db.users:
            raise ValueError("unknown user")
        u = self.db.users[user_id]
        pat = self._patterns(user_id)
        if target_duration is None:
            target_duration = min(u.time_availability, pat.get("preferred_duration", 30))

        scored = [(eid, self._compatibility(ex, u, pat)) for eid, ex in self.db.exercises.items()]
        scored.sort(key=lambda x: x[1], reverse=True)

        sel, rem = [], target_duration
        used_muscles = set()
        for eid, sc in scored:
            if rem <= 5:
                break
            ex = self.db.exercises[eid]
            overlap = len(set(ex.muscle_groups) & used_muscles)
            if overlap > 1 and sel:
                continue
            if ex.category == "cardio":
                base = min(15, rem // 2)
            elif ex.category == "strength":
                base = min(12, rem // 3)
            else:
                base = min(8, rem // 4)
            mult = {"beginner": 0.8, "intermediate": 1.0, "advanced": 1.2}.get(u.fitness_level, 1.0)
            dur = int(max(3, min(int(base * mult), rem)))
            if ex.category == "strength":
                if u.fitness_level == "beginner":
                    sets, reps = 2, "8-10"
                elif u.fitness_level == "intermediate":
                    sets, reps = 3, "10-12"
                else:
                    sets, reps = 3, "12-15"
            else:
                sets, reps = 1, f"{dur} minutes"
            sel.append({
                "exercise_id": eid,
                "name": ex.name,
                "category": ex.category,
                "duration": dur,
                "sets": sets,
                "reps": reps,
                "instructions": ex.instructions,
                "modifications": ex.modifications.get(u.fitness_level, ""),
                "calories_estimated": int(ex.calories_per_minute * dur),
                "muscle_groups": ex.muscle_groups
            })
            rem -= dur
            used_muscles.update(ex.muscle_groups)
            if len(sel) >= 8:
                break

        total_duration = sum(e["duration"] for e in sel)
        total_calories = sum(e["calories_estimated"] for e in sel)
        fscore = self._fitness_score(u)
        if fscore < 40:
            diff = "Easy"
        elif fscore < 70:
            diff = "Moderate"
        else:
            diff = "Challenging"
        return {
            "workout_id": _generate_id("workout"),
            "user_id": user_id,
            "date": datetime.date.today().isoformat(),
            "exercises": sel,
            "total_duration": int(total_duration),
            "estimated_calories": int(total_calories),
            "difficulty": diff,
            "focus_areas": sorted(list(used_muscles)),
            "user_fitness_score": float(round(fscore, 1)),
            "tips": self._tips(u, pat)
        }

    def _tips(self, u: UserProfile, pat: Dict) -> List[str]:
        tips = []
        f = self._fitness_score(u)
        if f < 50:
            tips.append("Prioritize technique; add intensity progressively.")
            tips.append("Schedule rest days to recover and adapt.")
        if pat.get("consistency", 0.0) < 50.0:
            tips.append("Use short, frequent sessions to build consistency.")
            tips.append("Set weekly micro-goals and track completions.")
        if u.bmi() > 25 and "weight_loss" in u.goals:
            tips.append("Pair cardio with strength; emphasize compound movements.")
        if u.age >= 50:
            tips.append("Include balance and mobility in each session.")
        if "back_pain" in u.medical_conditions:
            tips.append("Strengthen core; avoid heavy hip hinging early.")
        if not tips:
            tips.append("Hydrate well and log RPE to guide progression.")
        # dedupe, cap 3
        out, seen = [], set()
        for t in tips:
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) == 3:
                break
        return out

class ProgressTracker:
    def __init__(self, db: FitnessDatabase):
        self.db = db

    def metrics(self, user_id: str, days: int = 30) -> Dict:
        hist = self.db.history(user_id, days)
        if not hist:
            return {"error": "no_history", "days": days}
        total = len(hist)
        duration = sum(s.duration for s in hist)
        calories = sum(s.calories_burned for s in hist)
        avg_fb = float(np.mean([s.user_feedback for s in hist]))
        distinct = len(set(s.date for s in hist))
        consistency = distinct / float(days) * 100.0
        if len(hist) >= 2:
            k = min(7, len(hist))
            recent = hist[-k:]
            older = hist[:-k] if len(hist) > k else hist[:max(1, len(hist)//2)]
            recent_d = float(np.mean([s.duration for s in recent]))
            older_d = float(np.mean([s.duration for s in older])) if older else 0.0
            trend_d = _pct_change(recent_d, older_d) if older_d > 0 else 0.0

            recent_s = float(np.mean([s.user_feedback for s in recent]))
            older_s = float(np.mean([s.user_feedback for s in older])) if older else 0.0
            trend_s = _pct_change(recent_s, older_s) if older_s > 0 else 0.0
        else:
            trend_d = 0.0
            trend_s = 0.0

        all_ex = []
        for s in hist:
            all_ex.extend([e.get("name", "unknown") for e in s.exercises])
        variety = len(set(all_ex))

        weekly = {}
        for s in hist:
            wk = s.date.strftime("%Y-W%U")
            weekly.setdefault(wk, {"workouts": 0, "duration": 0, "calories": 0})
            weekly[wk]["workouts"] += 1
            weekly[wk]["duration"] += s.duration
            weekly[wk]["calories"] += s.calories_burned

        return {
            "total_workouts": total,
            "total_duration_hours": round(duration / 60.0, 2),
            "total_calories_burned": calories,
            "average_satisfaction": round(avg_fb, 2),
            "consistency_percentage": round(consistency, 1),
            "workout_frequency_per_week": round((total / days) * 7.0, 1),
            "duration_trend_percentage": round(trend_d, 1),
            "satisfaction_trend_percentage": round(trend_s, 1),
            "exercise_variety_count": variety,
            "weekly_breakdown": weekly,
            "analysis_period_days": days
        }

    def insights(self, user_id: str, days: int = 30) -> List[str]:
        m = self.metrics(user_id, days)
        if "error" in m:
            return ["Start logging sessions to unlock progress insights."]
        out = []
        if m["consistency_percentage"] >= 80:
            out.append("Excellent consistency—habits are solid.")
        elif m["consistency_percentage"] >= 50:
            out.append("Good consistency—try to add one more session weekly.")
        else:
            out.append("Focus on consistency—short sessions still count.")
        if m["duration_trend_percentage"] > 10:
            out.append("Endurance improving—longer sessions observed.")
        elif m["duration_trend_percentage"] < -10:
            out.append("Consider time-efficient intervals when busy.")
        if m["average_satisfaction"] >= 4.0:
            out.append("High enjoyment—keep leveraging preferred modalities.")
        elif m["average_satisfaction"] < 3.0:
            out.append("Vary exercises and adjust intensity for enjoyment.")
        if m["exercise_variety_count"] < 5:
            out.append("Increase variety to cover more muscle groups.")
        elif m["exercise_variety_count"] > 15:
            out.append("Great variety—helps prevent plateaus.")
        # cap 4
        return out[:4]

class PersonalizedFitnessCoach:
    def __init__(self):
        self.db = FitnessDatabase()
        self.reco = MLRecommendationEngine(self.db)
        self.progress = ProgressTracker(self.db)

    def create_user(self, data: Dict) -> str:
        uid = _generate_id("user")
        profile = UserProfile(
            user_id=uid,
            age=data["age"],
            gender=data["gender"],
            height=data["height"],
            weight=data["weight"],
            fitness_level=data["fitness_level"],
            goals=data.get("goals", ["general_fitness"]),
            medical_conditions=data.get("medical_conditions", []),
            preferred_workout_types=data.get("preferred_workout_types", []),
            available_equipment=data.get("available_equipment", []),
            time_availability=data.get("time_availability", 30),
            experience_years=data.get("experience_years", 0.0)
        )
        self.db.add_user(profile)
        return uid

    def recommend_workout(self, user_id: str, duration: Optional[int] = None) -> Dict:
        return self.reco.recommend(user_id, duration)

    def log_workout(self, user_id: str, exercises: List[Dict], duration: int,
                    calories_burned: int, user_feedback: int,
                    difficulty_rating: int = 5,
                    completion_status: str = "completed",
                    heart_rate_avg: Optional[int] = None,
                    notes: str = "") -> str:
        sid = _generate_id("session")
        s = WorkoutSession(
            session_id=sid,
            user_id=user_id,
            date=datetime.date.today(),
            exercises=exercises,
            duration=duration,
            calories_burned=calories_burned,
            difficulty_rating=difficulty_rating,
            completion_status=completion_status,
            user_feedback=user_feedback,
            heart_rate_avg=heart_rate_avg,
            notes=notes
        )
        self.db.log_session(s)
        return sid

    def progress_report(self, user_id: str, days: int = 30) -> Dict:
        return {
            "metrics": self.progress.metrics(user_id, days),
            "insights": self.progress.insights(user_id, days),
            "generated_at": _now_str()
        }

    def dashboard(self, user_id: str) -> Dict:
        if user_id not in self.db.users:
            return {"error": "user_not_found"}
        u = self.db.users[user_id]
        rep = self.progress_report(user_id, 30)
        next_w = self.recommend_workout(user_id)
        streak = self._streak(user_id)
        fav = self._favorite_exercise(user_id)
        return {
            "profile": {
                "user_id": u.user_id,
                "age": u.age,
                "gender": u.gender,
                "height_cm": u.height,
                "weight_kg": u.weight,
                "bmi": round(u.bmi(), 1),
                "bmr": int(u.bmr()),
                "fitness_level": u.fitness_level,
                "goals": u.goals
            },
            "progress": rep,
            "next_workout": next_w,
            "quick_stats": {
                "total_workouts": len(self.db.workouts.get(user_id, [])),
                "current_streak_days": streak,
                "favorite_exercise": fav
            }
        }

    def _streak(self, user_id: str) -> int:
        hist = self.db.history(user_id, 365)
        if not hist:
            return 0
        hist.sort(key=lambda s: s.date, reverse=True)
        streak = 0
        cur = datetime.date.today()
        for s in hist:
            if (cur - s.date).days == streak:
                streak += 1
            else:
                break
        return streak

    def _favorite_exercise(self, user_id: str) -> str:
        hist = self.db.history(user_id, 90)
        if not hist:
            return "None"
        count = {}
        for s in hist:
            for e in s.exercises:
                n = e.get("name", "Unknown")
                count[n] = count.get(n, 0) + 1
        return max(count, key=count.get) if count else "None"


def _demo_seed_users(coach: PersonalizedFitnessCoach) -> List[str]:
    seeds = [
        dict(age=27, gender="female", height=165, weight=60, fitness_level="beginner",
             goals=["weight_loss", "general_fitness"], available_equipment=[], time_availability=30,
             experience_years=0.5),
        dict(age=35, gender="male", height=180, weight=85, fitness_level="intermediate",
             goals=["muscle_gain", "strength"], available_equipment=["dumbbells", "bench"],
             time_availability=45, experience_years=3.0),
        dict(age=42, gender="female", height=170, weight=70, fitness_level="advanced",
             goals=["endurance", "general_fitness"], available_equipment=["yoga_mat"],
             time_availability=60, experience_years=8.0, medical_conditions=["back_pain"]),
    ]
    ids = []
    for d in seeds:
        uid = coach.create_user(d)
        ids.append(uid)
    return ids

def _demo_simulate_history(coach: PersonalizedFitnessCoach, user_ids: List[str], days: int = 14):
    for uid in user_ids:
        for i in range(days):
            if random.random() > 0.7:
                continue
            date = datetime.date.today() - datetime.timedelta(days=(days - 1 - i))
            plan = coach.recommend_workout(uid, duration=30)
            done = plan["exercises"][:random.randint(2, min(4, len(plan["exercises"])))]
            duration = random.randint(18, 40)
            calories = random.randint(140, 420)
            feedback = random.randint(3, 5)
            diff = random.randint(4, 8)
            status = random.choice(["completed", "completed", "partial"])
            sid = _generate_id("session")
            s = WorkoutSession(
                session_id=sid, user_id=uid, date=date,
                exercises=done, duration=duration, calories_burned=calories,
                difficulty_rating=diff, completion_status=status,
                user_feedback=feedback, heart_rate_avg=random.randint(115, 165)
            )
            coach.db.log_session(s)

def run_demo():
    print("🏋️ Personalized Fitness Coach (Demo)")
    print("=" * 64)
    coach = PersonalizedFitnessCoach()
    user_ids = _demo_seed_users(coach)
    print(f"[{_now_str()}] Created users:", ", ".join(user_ids))
    print(f"[{_now_str()}] Simulating history...")
    _demo_simulate_history(coach, user_ids)
    for uid in user_ids:
        print("\n" + "-" * 64)
        print("User:", uid)
        dash = coach.dashboard(uid)
        prof = dash["profile"]
        print(f"Profile -> Age {prof['age']}, BMI {prof['bmi']}, Level {prof['fitness_level']}, BMR {prof['bmr']}")
        print("Goals:", ", ".join(prof["goals"]))
        if "error" not in dash["progress"]["metrics"]:
            m = dash["progress"]["metrics"]
            print(f"Progress -> Workouts {m['total_workouts']}, Hours {m['total_duration_hours']}, "
                  f"Calories {m['total_calories_burned']}, Consistency {m['consistency_percentage']}%")
            print("Insights:")
            for t in dash["progress"]["insights"]:
                print(" -", t)
        print("Next Workout:")
        nw = dash["next_workout"]
        print(f"  Duration {nw['total_duration']} min | Difficulty {nw['difficulty']} | Calories ~{nw['estimated_calories']}")
        print("  Focus:", ", ".join(nw["focus_areas"]))
        print("  Sample Exercises:")
        for i, ex in enumerate(nw["exercises"][:3], 1):
            print(f"   {i}. {ex['name']} - {ex['duration']} min | {ex['category']}")
        print("Tips:")
        for t in nw["tips"]:
            print(" -", t)
        qs = dash["quick_stats"]
        print(f"Quick Stats -> Streak {qs['current_streak_days']} day(s), Favorite {qs['favorite_exercise']}")
    print("\n" + "=" * 64)
    print("Demo complete. Extend with persistence, API, or UI as needed.")

def cli():
    coach = PersonalizedFitnessCoach()
    print("Personalized Fitness Coach CLI")
    print("Commands: create, plan, log, report, dash, demo, exit")
    current_user: Optional[str] = None
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "exit":
            print("Bye.")
            break
        if cmd == "demo":
            run_demo()
            continue
        if cmd == "create":
            try:
                age = int(input("age: "))
                gender = input("gender (male/female): ").strip().lower()
                height = float(input("height_cm: "))
                weight = float(input("weight_kg: "))
                level = input("fitness_level (beginner/intermediate/advanced): ").strip().lower()
                goals = [g.strip().lower() for g in input("goals (comma): ").split(",") if g.strip()]
                equip = [e.strip().lower() for e in input("equipment (comma): ").split(",") if e.strip()]
                time_avail = int(input("time_availability_minutes: "))
                exp = float(input("experience_years: "))
                uid = coach.create_user(dict(
                    age=age, gender=gender, height=height, weight=weight,
                    fitness_level=level, goals=goals, available_equipment=equip,
                    time_availability=time_avail, experience_years=exp
                ))
                current_user = uid
                print("user_id:", uid)
            except Exception as e:
                print("error:", e)
            continue
        if cmd == "plan":
            if not current_user:
                print("no current user. run 'create' first.")
                continue
            try:
                dur = input("target_duration (blank=auto): ").strip()
                dur = int(dur) if dur else None
                rec = coach.recommend_workout(current_user, dur)
                print(json.dumps(rec, indent=2))
            except Exception as e:
                print("error:", e)
            continue
        if cmd == "log":
            if not current_user:
                print("no current user. run 'create' first.")
                continue
            try:
                duration = int(input("session_duration_min: "))
                calories = int(input("calories_burned: "))
                feedback = int(input("user_feedback (1-5): "))
                sid = coach.log_workout(
                    current_user, exercises=[], duration=duration,
                    calories_burned=calories, user_feedback=feedback
                )
                print("session_id:", sid)
            except Exception as e:
                print("error:", e)
            continue
        if cmd == "report":
            if not current_user:
                print("no current user. run 'create' first.")
                continue
            try:
                days = int(input("days (default 30): ") or "30")
                rep = coach.progress_report(current_user, days)
                print(json.dumps(rep, indent=2))
            except Exception as e:
                print("error:", e)
            continue
        if cmd == "dash":
            if not current_user:
                print("no current user. run 'create' first.")
                continue
            try:
                dash = coach.dashboard(current_user)
                print(json.dumps(dash, indent=2))
            except Exception as e:
                print("error:", e)
            continue
        print("unknown command.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        run_demo()
        print("\nEntering interactive CLI. Type 'exit' to quit.")
        try:
            cli()
        except KeyboardInterrupt:
            print("\nBye.")
