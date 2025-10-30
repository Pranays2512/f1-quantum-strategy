import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import Response
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import base64

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def rdp(points, epsilon):
    """Ramer-Douglas-Peucker algorithm for track simplification"""
    
    def perpendicular_distance(pt, line_start, line_end):
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        # Fix for NumPy 2.0 - ensure 1D arrays for cross product
        v1 = line_end - line_start
        v2 = line_start - pt
        return np.abs(v1[0] * v2[1] - v1[1] * v2[0]) / np.linalg.norm(v1)

    def rdp_core(pts, epsilon):
        if len(pts) < 3:
            return pts
        start, end = pts[0], pts[-1]
        dmax = 0
        index = 0
        for i in range(1, len(pts)-1):
            d = perpendicular_distance(pts[i], start, end)
            if d > dmax:
                index, dmax = i, d
        if dmax > epsilon:
            res1 = rdp_core(pts[:index+1], epsilon)
            res2 = rdp_core(pts[index:], epsilon)
            return np.vstack((res1[:-1], res2))
        else:
            return np.array([start, end])
    
    return rdp_core(points, epsilon)

def analyze_track_image(img_bytes: bytes, epsilon: float):
    """Analyze track image and extract polygon representation"""
    # 1. Decode image
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None, None

    # 2. Preprocess and find track contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not cnts:
        return None, None, None, None
    
    cnt = max(cnts, key=cv2.contourArea)  # Assume biggest contour is the track
    curve = np.squeeze(cnt)
    
    if curve.ndim != 2 or curve.shape[0] < 3:
        return None, None, None, None

    # 3. Simplify with RDP algorithm
    epsilon_pixels = epsilon * cv2.arcLength(cnt, True)
    simplified = rdp(curve, epsilon_pixels)
    
    # Always close the loop
    if simplified[0].tolist() != simplified[-1].tolist():
        simplified = np.vstack([simplified, simplified[0]])

    # 4. Compute segment lengths and corner angles
    segments = []
    total_len = 0
    angles = []
    
    for i in range(len(simplified) - 1):
        p1 = simplified[i]
        p2 = simplified[i+1]
        length = np.linalg.norm(p2 - p1)
        total_len += length
        segments.append({
            "segment_id": i,
            "start": p1.tolist(),
            "end": p2.tolist(),
            "length": float(length),
            "corner_angle": None
        })
    
    # Calculate angles between consecutive segments
    for i in range(1, len(segments)):
        v1 = np.array(segments[i-1]["end"]) - np.array(segments[i-1]["start"])
        v2 = np.array(segments[i]["end"]) - np.array(segments[i]["start"])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(np.clip(dot / norm, -1, 1)) * 180 / np.pi
        angles.append(angle)
        segments[i]["corner_angle"] = float(angle)
    
    if segments:
        segments[0]["corner_angle"] = angles[-1] if angles else 0

    # 5. Visualization (simplified path with correct orientation)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(simplified[:, 0], simplified[:, 1], 'r-', lw=2, marker='o', 
            markersize=4, label='Simplified Track Path')

    # Set correct orientation and aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    # Ensure the path is fully visible
    ax.relim()
    ax.autoscale_view()

    # Clean plot
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title('Track Analysis', fontsize=14, pad=10)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, 
                transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)
    vis_data = base64.b64encode(buf.read()).decode()
    img_url = f"data:image/png;base64,{vis_data}"

    # 6. Collect statistics
    avg_angle = float(np.mean(angles)) if angles else 0
    result = {
        "segments": [{
            "segment_id": s["segment_id"], 
            "start": s["start"],
            "end": s["end"],
            "length": s["length"], 
            "corner_angle": s["corner_angle"] or 0
        } for s in segments],
        "statistics": {
            "num_segments": len(segments),
            "total_length": float(total_len),
            "average_angle": avg_angle
        },
        "visualization": img_url
    }
    
    return result, segments, curve, simplified

@app.get("/")
def read_root():
    return {"status": "F1 Track Analysis Backend", "version": "1.0"}

@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), epsilon: float = 0.01):
    """Analyze uploaded track image and return polygon segments"""
    img_bytes = file.file.read()
    result, _, _, _ = analyze_track_image(img_bytes, epsilon)
    
    if not result:
        return JSONResponse(
            status_code=400, 
            content={"detail": "No closed contour found in image. Ensure track has clear boundaries."}
        )
    
    return {"data": result}

@app.post("/api/analyze/csv")
def analyze_csv(file: UploadFile = File(...), epsilon: float = 0.01):
    """Analyze track and return results as CSV"""
    img_bytes = file.file.read()
    result, segments, _, _ = analyze_track_image(img_bytes, epsilon)
    
    if not result or not segments:
        return JSONResponse(
            status_code=400, 
            content={"detail": "No closed contour found in image."}
        )
    
    # Create CSV
    df = pd.DataFrame([{
        "Segment": s["segment_id"],
        "Length (px)": s["length"],
        "Corner Angle (deg)": s["corner_angle"] or 0
    } for s in segments])
    
    csv_bytes = df.to_csv(index=False).encode()
    
    return Response(
        content=csv_bytes, 
        media_type="text/csv", 
        headers={"Content-Disposition": "attachment; filename=track_segments.csv"}
    )

@app.post("/api/strategy/analyze")
def analyze_strategy(race_data: dict):
    """
    Mock endpoint for AI strategy analysis
    Replace this with your actual AI model
    """
    import time
    start_time = time.time()
    
    # Mock AI response - replace with your actual AI logic
    response = {
        "immediate_action": {
            "action": "OVERTAKE",
            "priority": "HIGH",
            "confidence": 85,
            "reasoning": "Gap to P4 is optimal for overtake attempt in DRS zone",
            "execute_immediately": True
        },
        "pit_stop_recommendation": {
            "recommendation": "Pit on lap 35 for medium tyres",
            "optimal_lap": 35,
            "tyre_compound": "MEDIUM",
            "confidence": 78,
            "expected_time_impact": -2.3
        },
        "pace_strategy": {
            "pace_mode": "PUSH",
            "lap_time_target": "1:26.500",
            "fuel_saving_required": False
        },
        "overtaking_opportunities": [{
            "target_car": "car_3",
            "current_position": 4,
            "probability": 85,
            "recommended_zone": "DRS Zone 1",
            "recommendation": "Attack into Turn 1 with DRS advantage"
        }],
        "response_time_ms": int((time.time() - start_time) * 1000)
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)