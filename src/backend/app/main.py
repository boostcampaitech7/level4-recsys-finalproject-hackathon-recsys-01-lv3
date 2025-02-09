from fastapi import FastAPI
from app.api.endpoints import model

app = FastAPI(title="presciptive promo API")

# API 엔드포인트 추가
app.include_router(model.router)

@app.get("/")
def root():
    return {"message": "Jeonse Price Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)