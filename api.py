from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.allocator import allocate_bed


app = FastAPI(
    title="Hospital Resource Allocation API",
    description="API for hospital bed and resource allocation",
    version="1.0"
)


# ==========================================
# REQUEST MODEL
# ==========================================

class AllocationRequest(BaseModel):

    ward_type: str = Field(
        ...,
        description="Required ward type"
    )

    stay_days: int = Field(
        1,
        ge=1,
        description="Expected patient stay in days"
    )

    oxygen: bool = False

    ventilator: bool = False

    isolation: bool = False

    patient_id: str = "P999"


# ==========================================
# HOME
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Hospital Resource Allocation API",
        "status": "running"
    }


# ==========================================
# ALLOCATE BED
# ==========================================

@app.post("/allocate")
def allocate(request: AllocationRequest):

    requirements = {
        "ward_type": request.ward_type.strip(),

        "stay_days": request.stay_days,

        "oxygen": request.oxygen,

        "ventilator": request.ventilator,

        "isolation": request.isolation,

        "patient_id": request.patient_id
    }

    print("\n==============================")
    print("FASTAPI ALLOCATION REQUEST")
    print("==============================")
    print(requirements)

    result = allocate_bed(requirements)

    print("\nALLOCATION RESULT")
    print(result)
    print("==============================\n")

    return result