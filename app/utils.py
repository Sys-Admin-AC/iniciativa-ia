from datetime import datetime
from zoneinfo import ZoneInfo

def get_now():
    """Retorna el datetime actual en la zona horaria de Managua, como objeto naive (sin tzinfo) 
    para mantener compatibilidad con la base de datos actual."""
    return datetime.now(ZoneInfo("America/Managua")).replace(tzinfo=None)
