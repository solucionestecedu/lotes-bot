# Cómo probar Lotes-Bot (paso a paso)

## Prueba 1 · En el navegador, sin IA (5 minutos)

Sirve para ver el flujo completo antes de poner llaves.

1. Corre el bot en tu computadora (ver README) y abre `http://localhost:8080/prueba`.
2. Pega esto en el chat y envía. La primera línea es la coordenada del vértice Mo.1; las demás son los rumbos de Chiltiupán:

   ```
   inicio 450142.118, 265697.441
   S 82°40'17.03" E 82.87
   S 83°07'19.69" E 27.21
   S 82°31'14.44" E 47.62
   S 81°55'50.86" E 9.04
   S 82°30'14.09" E 54.23
   S 83°05'42.05" E 49.39
   S 82°52'11.39" E 39.50
   S 82°41'30.40" E 80.09
   S 03°25'56.11" O 20.22
   S 77°53'06.59" E 21.62
   S 08°39'28.54" O 13.80
   N 78°55'02.68" O 37.01
   N 82°27'55.42" O 62.94
   N 83°31'13.52" O 40.80
   N 79°33'57.88" O 73.69
   N 81°32'09.82" O 113.39
   N 78°25'25.79" O 72.92
   N 21°59'09.21" O 24.94
   ```

   **Debe decir:** 18 vértices, 11,698.16 m², cierre 0.001 m.
3. Escribe `relieve` → desnivel de unos 138 m. *(Necesita internet.)*
4. Escribe `lotes` → 26 lotes.
5. Escribe `lámina` → llegan el PDF, el KML y el DXF.

## Prueba 2 · Con Gemini leyendo fotos

1. Saca una llave en <https://aistudio.google.com/apikey> y ponla en `GEMINI_API_KEY`.
2. Reinicia el bot. En `/health` debe decir `"modo": "GeminiLLM"`.
3. En `/prueba`, adjunta (📎) la foto del cuadro de rumbos y escribe "hola, este es el plano".
4. Revisa:
   - [ ] ¿Leyó todos los tramos? Compara con el plano.
   - [ ] Si el plano cierra, ¿lo dijo? Si no cierra, ¿te señaló el tramo sospechoso?
   - [ ] Si el plano no trae coordenadas, ¿te pidió la ubicación? Usa 📍 o escribe la coordenada del vértice 1.
   - [ ] Pide el relieve, los lotes y la lámina en palabras normales ("hazme los lotes de 8 por 20").
   - [ ] ¿Te recordó que es anteproyecto y no sirve para trámite?

Prueba también el **plano B**. No cierra por 7.70 m. El bot debe frenar y pedirte revisar, no inventar una corrección.

### Medir la lectura con números

```bash
python evaluar_lectura.py foto_del_cuadro.jpg tests/datos/plano_a.txt
```

Marca cada tramo como OK o MAL y te dice si el cierre habría atrapado el error. Hazlo con varias fotos: derechas,
chuecas, con poca luz. Así sabes cuánto confiar en la lectura antes de mostrarlo a clientes.

## Prueba 3 · En WhatsApp con el número de prueba de Meta

1. Publica en Cloud Run y conecta el webhook (README, secciones 2 y 3).
2. Desde tu celular, escríbele "hola" al número de prueba.
3. Repite la prueba 2, ahora en WhatsApp. Revisa también:
   - [ ] ¿La foto tomada con el celular se lee igual que la escaneada?
   - [ ] ¿La ubicación compartida de WhatsApp ubica bien el terreno?
   - [ ] ¿Llegan los tres archivos? ¿Se abren el KML en Google Earth y el DXF en AutoCAD?
   - [ ] ¿El PDF impreso en tabloide al 100 % mide bien con escalímetro?
4. `reiniciar` borra la conversación.

## Si algo falla

| Síntoma | Qué revisar |
|---|---|
| Meta no verifica el webhook | Que `META_WEBHOOK_VERIFY_TOKEN` sea idéntico en Meta y en Cloud Run |
| El bot no contesta en WhatsApp | Logs de Cloud Run; `META_APP_SECRET` correcto; tu número en `NUMEROS_PERMITIDOS` |
| "No pude descargar las alturas" | Salida a internet del servidor; vuelve a intentar |
| Lee mal el cuadro | Foto más cerca, derecha y con luz; o pega los rumbos como texto |
| Se le olvida la plática | Se reinició el servidor (pasa si no hay uso por un rato). Reenvía el plano |
