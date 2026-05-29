from datetime import date
from typing import List
from pydantic import BaseModel

class KeyPerformanceIndicator(BaseModel):
    title: str
    value: float
    trend: float
    
class CustomerChartData(BaseModel):
    date: str | date
    activeAccounts: int
    newCustomers: int
    
class CustomerData(BaseModel):
    id: str
    segment: str
    orderCount: int
    orderAmount: float
    joinedDate: str | date

class PaginationMetadata(BaseModel):
    currentPage: int
    perPage: int
    totalPage: int
    totalData: int
    allSegments: List[str]

class KPIResponse(BaseModel):
    status: str = "success"
    data: List[KeyPerformanceIndicator]

class ChartDataResponse(BaseModel):
    status: str = "success"
    data: List[CustomerChartData]

class CustomerDataResponse(BaseModel):
    status: str = "success"
    metadata: PaginationMetadata
    data: List[CustomerData]