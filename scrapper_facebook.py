import os
from playwright.sync_api import sync_playwright
from anthropic import Anthropic
import base64
from typing import Dict, Optional
from enum import Enum
import json

class ClaudeModel(Enum):
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-sonnet-20240229"
    OPUS = "claude-3-opus-20240229"

class CostTracker:
    # Precios actualizados por 1M tokens (MTok)
    PRICING = {
        ClaudeModel.HAIKU.value: {
            "input": 0.80,        # $0.80 por MTok de input
            "output": 4.00,       # $4.00 por MTok de output
            "cache_write": 1.00,  # $1.00 por MTok de cache write
            "cache_read": 0.08    # $0.08 por MTok de cache read
        },
        ClaudeModel.SONNET.value: {
            "input": 3.00,
            "output": 15.00,
            "cache_write": 3.75,
            "cache_read": 0.30
        },
        ClaudeModel.OPUS.value: {
            "input": 15.00,
            "output": 75.00,
            "cache_write": 18.75,
            "cache_read": 1.50
        }
    }
    
    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int,
                       cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> Dict[str, float]:
        """
        Calcula el costo detallado de una llamada a la API basado en los tokens utilizados.
        """
        if model not in CostTracker.PRICING:
            raise ValueError(f"Modelo {model} no encontrado en la tabla de precios")
            
        pricing = CostTracker.PRICING[model]
        
        # Convertir tokens a millones (MTok)
        input_mtok = input_tokens / 1_000_000
        output_mtok = output_tokens / 1_000_000
        cache_write_mtok = cache_write_tokens / 1_000_000
        cache_read_mtok = cache_read_tokens / 1_000_000
        
        # Calcular costos individuales
        input_cost = input_mtok * pricing["input"]
        output_cost = output_mtok * pricing["output"]
        cache_write_cost = cache_write_mtok * pricing["cache_write"]
        cache_read_cost = cache_read_mtok * pricing["cache_read"]
        
        # Calcular costo total
        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost
        
        return {
            "model": model,
            "costs": {
                "input": round(input_cost, 7),
                "output": round(output_cost, 7),
                "cache_write": round(cache_write_cost, 7),
                "cache_read": round(cache_read_cost, 7),
                "total": round(total_cost, 7)
            },
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "cache_write": cache_write_tokens,
                "cache_read": cache_read_tokens,
                "total": input_tokens + output_tokens + cache_write_tokens + cache_read_tokens
            }
        }

class VehicleMarketplaceAnalyzer:
    def __init__(self, anthropic_api_key):
        """
        Inicializa el analizador con la API key de Anthropic
        """
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.last_request_cost = None
        
    def take_screenshot(self, url, output_file="screenshot.png", width=1920, height=1080, clip_width=600):
        """
        Toma una captura de pantalla de Facebook Marketplace (u otro) utilizando cookies pre-cargadas.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                
                # Leer cookies pre-cargadas desde un archivo JSON
                cookies_file = "facebook_cookies.json"
                cookies = []
                if os.path.exists(cookies_file):
                    try:
                        with open(cookies_file, "r") as f:
                            cookies = json.load(f)
                    except Exception as e:
                        print(f"Error al cargar cookies desde {cookies_file}: {e}")
                        cookies = []
                else:
                    print(f"No se encontró el archivo de cookies '{cookies_file}'. Asegúrate de generarlo previamente.")
                
                # Crear el contexto sin usar 'cookies=' porque las versiones recientes de Playwright
                # ya no aceptan ese parámetro en new_context()
                context = browser.new_context(
                    viewport={'width': width, 'height': height}
                )
                
                # Agregar cookies manualmente si existen
                if cookies:
                    context.add_cookies(cookies)
                
                page = context.new_page()
                # Navegar a la página esperando que el DOM se cargue
                page.goto(url, wait_until='domcontentloaded', timeout=20000)
                
                # Breve pausa para que se rendericen elementos básicos
                page.wait_for_timeout(2000)
                
                # Intentar esperar un selector opcional
                try:
                    page.wait_for_selector('text=Descripción del vendedor', timeout=5000)
                except Exception:
                    print("No se encontró el selector específico, continuando de todos modos...")
                
                # Tomar el screenshot con el área de recorte deseada
                page.screenshot(
                    path=output_file,
                    clip={"x": width - clip_width, "y": 0, "width": clip_width, "height": height}
                )
                
                context.close()
                browser.close()
            return True
        except Exception as e:
            print(f"Error tomando screenshot: {str(e)}")
            return False

    def encode_image(self, image_path):
        """
        Codifica la imagen en base64
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error codificando imagen: {str(e)}")
            return None

    def analyze_vehicle_listing(self, image_path, phone_number):
        """
        Analiza la imagen del listado del vehículo usando Claude y rastrea los costos.
        Se sustituye la sección de teléfono por el phone_number que recibe como parámetro.
        """
        try:
            image_data = self.encode_image(image_path)
            if not image_data:
                return None

            prompt = f"""Analiza esta imagen y extrae la información siguiendo estas reglas estrictas:

1. PRECIO:
- Extrae el precio numérico exacto de la imagen
- Elimina cualquier punto o coma del número
- Multiplica ese número por 1.04 para añadir el 4%
- Formatea el resultado final con puntos como separadores de miles

2. IDENTIFICACIÓN:
- Extrae la marca y modelo exactos
- Si hay variante específica del modelo, inclúyela

Usa EXACTAMENTE este formato. Cuando no encuentres un dato, omite el campo completo, desde la parte de las especificaciones del vehiculo hasta "📍Edificio Access Point - Av. Las Palmas (cita previa)" que es donde termina el mensaje:

🚘 [MARCA] [MODELO]
[INCLUIR SOLO LOS CAMPOS QUE ESTÉN PRESENTES EN LA IMAGEN, CADA UNO EN UNA NUEVA LÍNEA Y CON EL PREFIJO ➖]
➖Precio: $[PRECIO DE LA IMAGEN + 4% en formato numérico con puntos como separadores de miles]
➖Motor: [MOTOR si está disponible]
➖Modelo: [AÑO]
➖Kilómetros: [KILOMETRAJE en formato numérico]
➖Ubicación: [UBICACIÓN]
➖Otros: [INCLUIR: estado del vehículo, documentos al día, características especiales de equipamiento]

Llamada celular / WhatsApp 
📲 {phone_number}
Consigue tu vehículo ideal con @autos_st, contáctanos si deseas vender tu vehículo con nosotros.
Recuerda que con nosotros puedes sacar tu crédito fácil y rápido. Aprobación en 3 horas, certificamos las mejores tasas del mercado💸
📍Edificio Access Point - Av. Las Palmas (cita previa)

REGLAS DE VALIDACIÓN:
1. El precio DEBE ser un número
2. El kilometraje DEBE ser un número sin puntos ni texto adicional
3. El año DEBE ser un número de 4 dígitos entre 1900 y 2024
4. La ubicación DEBE ser una ciudad o zona específica

INSTRUCCIONES IMPORTANTES:
1. Solo incluir los campos con ➖ que estén presentes en la imagen
2. Todo el texto después de "Llamada celular / WhatsApp" debe incluirse EXACTAMENTE como está mostrado arriba
3. No modificar ningún emoji o formato
4. No agregar información adicional al final
"""
            model = ClaudeModel.HAIKU.value
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}}
                    ]
                }]
            )
            self.last_request_cost = CostTracker.calculate_cost(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            if isinstance(response.content, list) and len(response.content) > 0:
                first_block = response.content[0]
                return {
                    'analysis': first_block.text.strip(),
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            else:
                print("Respuesta no reconocida:", response.content)
                return None
        except Exception as e:
            print(f"Error analizando imagen con Claude: {str(e)}")
            return None

    def get_last_request_cost(self) -> Optional[Dict]:
        """
        Retorna la información detallada de costo de la última solicitud
        """
        return self.last_request_cost

def main():
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    url = "https://www.facebook.com/marketplace/item/1336536434139458/"
    
    analyzer = VehicleMarketplaceAnalyzer(ANTHROPIC_API_KEY)
    screenshot_path = "vehicle_listing.png"
    
    if analyzer.take_screenshot(url, screenshot_path):
        phone_number = "+57 3009998888"
        result = analyzer.analyze_vehicle_listing(screenshot_path, phone_number)
        if result:
            print("\nAnálisis del vehículo:")
            print(result['analysis'])
            
            print("\nTokens utilizados en esta solicitud:")
            print(f"Input tokens: {result['input_tokens']:,}")
            print(f"Output tokens: {result['output_tokens']:,}")
            
            cost_info = analyzer.get_last_request_cost()
            if cost_info:
                print("\nInformación detallada de costos:")
                print(f"Modelo: {cost_info['model']}")
                print("\nCostos:")
                for category, amount in cost_info['costs'].items():
                    print(f"  {category.capitalize()}: ${amount:.7f}")
                print("\nTokens utilizados:")
                for category, count in cost_info['tokens'].items():
                    print(f"  {category.capitalize()}: {count:,}")
        else:
            print("No se pudo analizar la imagen")
    else:
        print("No se pudo tomar la captura de pantalla")

if __name__ == "__main__":
    main()
