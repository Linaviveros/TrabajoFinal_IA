1. Activar el entorno virtual
Antes de instalar dependencias o ejecutar servicios, se debe activar el entorno virtual del proyecto. Esto permite aislar los paquetes requeridos y evitar conflictos con otras instalaciones de Python.

Ubícate en la carpeta principal del proyecto usando la terminal.

Ejecuta el siguiente comando en Windows:


.\venv\Scripts\activate
Cuando el entorno esté activo, verás el nombre del entorno al inicio de la línea de comandos.​​

Nota: Si estás en Linux o MacOS, usa source venv/bin/activate.

2. Instalar las dependencias
Con el entorno virtual activo, instala todas las dependencias necesarias para el proyecto utilizando el archivo requirements.txt. Este archivo contiene la lista de paquetes que tu proyecto necesita.

text
pip install -r requirements.txt
Este comando asegurará que tanto el detector de cámara como el API trabajen con las versiones específicas requeridas.​​

3. Iniciar la cámara para la detección
Para activar el sistema de detección mediante la cámara, usa el siguiente comando:


python -m detector.live_camera
Este iniciará el módulo que utiliza la cámara para realizar la detección en tiempo real según el código de tu proyecto.​

4. Iniciar el servicio API
Para lanzar el API del proyecto, que probablemente utiliza FastAPI y Uvicorn, ejecuta este comando:


python -m uvicorn api.main:app --reload
Esto iniciará el servidor en modo recarga automática, ideal para desarrollo y pruebas. El API se encontrará disponible en el puerto y la dirección configurada en tu script principal bajo api.main:app.​

​

Recomendaciones finales
Si tienes problemas con paquetes o versiones, verifica que activaste correctamente el entorno virtual antes de instalar dependencias.
Como por ultima recomendacion se tiene que mandar los comandos de activacion de camara y activacion del servicio por terminales diferentes.

Ejecuta siempre los comandos desde la raíz del proyecto.

Si el profesor utiliza otro sistema operativo, debe adaptar el comando de activación del entorno virtual según corresponda.

Esto garantiza que el proyecto funcione igual de correcto y libre de errores de dependencias o conflictos.
