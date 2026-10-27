# Parcial 2 – Inteligencia Artificial (Grupo 09)

**Estudiante:** Cristian David Zuleta Restrepo (trabajo individual)
**Archivo asignado:** `grupo_09.csv` – Educación, estudiantes de Cartago
**Objetivo a predecir:** `desercion` (1 = desertó, 0 = continúa)
**Entradas:** `asistencia_pct` (x1) y `promedio_notas` (x2)

## Contenido del repositorio

```
parcial2/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── src/
│   └── parcial2.py
└── data/
    ├── grupo_09.csv                  # CSV original
    ├── grupo_09_limpio.csv           # CSV limpio
    ├── grafico1_dispersion.png
    ├── grafico2_boxplot_notas.png
    ├── curva_aprendizaje.png
    └── frontera_decision.png
```

## Tarea 1. Entorno y comandos para reproducirlo

Entorno: Python 3.12, NumPy, Pandas, Matplotlib y Seaborn en un contenedor Docker
(`container_name: parcial2_ia`). La carpeta del proyecto se monta en `/app`.

```bash
# En WSL, desde la carpeta del curso
mkdir -p parcial2/src parcial2/data
cd parcial2
# copiar grupo_09.csv dentro de data/

docker ps                              # si hay otro contenedor encendido: docker stop <nombre>
docker compose up -d --build           # construye la imagen y levanta parcial2_ia
docker exec -it parcial2_ia bash       # entrar al contenedor
cd src
python parcial2.py                     # ejecuta todo el análisis
exit
docker compose down                    # apagar al terminar
```

## Tarea 2. Carga e inspección

El script `src/parcial2.py` imprime en consola:

- **Primeras 5 filas:** E001 a E005 (Sistemas y Administración).
- **Forma (shape):** (84, 5).
- **Tipos de datos:** `estudiante_id`, `programa` y `asistencia_pct` son texto;
  `promedio_notas` es `float64` y `desercion` es `int64`.
- **Nulos por columna:** 3 en `promedio_notas`; 0 en las demás.
- **Filas duplicadas:** 4.

**Pregunta clave: ¿alguna columna numérica tiene un tipo que no debería?**
Sí: `asistencia_pct` aparece como texto en lugar de número. Ocurre porque tres
celdas incluyen el símbolo `%` (E023 = `79%`, E053 = `90%` y E061 = `41%`); al
encontrar texto en la columna, Pandas lee todos sus valores como texto.

## Tarea 3. Calidad y limpieza

Se encontraron **4 problemas**:

| # | Problema | ¿Cómo se detectó? | Decisión | ¿Por qué? |
|---|---|---|---|---|
| 1 | Texto con `%` en `asistencia_pct` (E023, E053, E061) | `df.dtypes` mostró texto y `df['asistencia_pct'].str.contains('%')` encontró 3 filas | **Convertir:** quitar el `%` y pasar a número | Los valores son válidos (79, 90 y 41); eliminarlos desperdiciaría 3 registros y el perceptrón necesita números |
| 2 | 4 filas duplicadas (E016, E029, E037, E079) | `df.duplicated().sum()` = 4 y `df[df.duplicated(keep=False)]` | **Eliminar** las repetidas, conservando la primera aparición | Un estudiante contado dos veces pesa doble en la pérdida y en la media y la desviación |
| 3 | Valor imposible: `promedio_notas = 82.0` (E051); la escala es 0 a 5 | `df['promedio_notas'] > 5` | **Eliminar** la fila | No se puede saber cuál era la nota real (podría ser 2.8, 8.2 o una nota de 0 a 100); inventarla metería ruido |
| 4 | 3 valores faltantes en `promedio_notas` (E026, E028, E078) | `df.isnull().sum()` y `df[df.isnull().any(axis=1)]` | **Eliminar** esas filas | La nota es una de las dos entradas y se relaciona mucho con la deserción (75 % de deserción con notas menores a 2.5); imputarla inventaría información |

También se revisó `programa`: tiene 5 valores bien escritos (Administración,
Contaduría, Gestión Empresarial, Mecatrónica y Sistemas), sin variantes.

**Filas:** 84 antes → **76** después (se eliminaron 4 duplicadas, 1 imposible y 3 con
nulos).
**Clases:** antes 45 en clase 0 y 39 en clase 1; después **43 en clase 0 y 33 en clase 1**.
Resultado guardado en `data/grupo_09_limpio.csv`.

**Pregunta clave: ¿qué valor fue el más sospechoso?**
El `82.0` de `promedio_notas` (E051), diez veces más grande que el máximo posible.

| | Media de `promedio_notas` | Desviación |
|---|---|---|
| Antes de corregirlo | 4.1753 | 8.9825 |
| Después de corregirlo | 3.1513 | 1.0023 |

Si no se hubiera corregido, la desviación se habría multiplicado por casi 9. Al
estandarizar, el 82 quedaría en +8.66 desviaciones y las demás notas (1.5 a 4.9)
se comprimirían entre -0.30 y +0.08, casi como una constante, en vez de repartirse
entre -1.65 y +1.75. El perceptrón no podría aprovechar las notas. Se comprobó
entrenando con el 82 sin corregir: `w2` queda en 0.0223 (casi cero, y con el
signo equivocado) y la exactitud baja a 72.73 %, frente a 92.11 % con los datos limpios.

## Tarea 4. Visualización con Seaborn

**Gráfico 1 – Dispersión de asistencia y notas según deserción**
(`data/grafico1_dispersion.png`). Los estudiantes que desertaron (rojo) se concentran
abajo a la izquierda (asistencia baja y notas bajas) y los que continúan (verde) arriba
a la derecha. La asistencia promedio es 60.85 % en los desertores y 80.12 % en
quienes continúan, y la nota promedio es 2.60 contra 3.57. Conclusión: ambas
variables se relacionan con la deserción y los dos grupos se separan en diagonal, pero se
mezclan en el centro, así que una línea recta los separa de forma aproximada.

**Gráfico 2 – Boxplot del promedio de notas según deserción**
(`data/grafico2_boxplot_notas.png`). La mediana de notas es 3.6 en quienes continúan y
2.2 en quienes desertaron. Los cuerpos de las cajas se traslapan entre 2.9 y 3.5
(el cuartil 3 de los desertores es 3.5 y el cuartil 1 de los que continúan es 2.9).
Conclusión: la nota diferencia bien a los dos grupos, pero por sí sola no basta;
algunos desertores tienen notas altas (hasta 4.9).

## Tarea 5. Perceptrón desde cero (NumPy)

Implementado en `src/parcial2.py` sin Scikit-Learn: estandarización de X (se guardan
la media [71.75, 3.1513] y la desviación [17.8201, 1.0023]), sigmoide,
probabilidad `sigmoide(X @ w + b)`, pérdida logarítmica, exactitud y descenso del
gradiente con tasa de aprendizaje 0.1 durante **500 épocas**.

| Resultado | Valor |
|---|---|
| Pesos iniciales | w1 = 0, w2 = 0, b = 0 |
| Pesos finales | w1 = -2.6678, w2 = -2.4402, b = -0.3884 |
| Pérdida inicial → final | 0.6931 → 0.2226 |
| Exactitud final | 92.11 % |

Gráficos: `data/curva_aprendizaje.png` (la pérdida baja de forma continua en las 500
épocas) y `data/frontera_decision.png` (frontera en las unidades originales).

## Tarea 6. Interpretación de resultados

**1. Signo de w1 y w2.** Ambos son negativos (w1 = -2.6678 para la asistencia y
w2 = -2.4402 para las notas). Como la sigmoide da la probabilidad de deserción, a
mayor asistencia y a mayores notas, menor probabilidad de desertar. Tiene sentido
para Cartago: los estudiantes que faltan más y rinden menos son los que abandonan.
Como las entradas están estandarizadas, los pesos se pueden comparar y son
parecidos: asistencia y notas pesan casi igual.

**2. ¿Son linealmente separables?** Aproximadamente, pero no perfectamente. En la
frontera quedan **6 de 76 registros del lado equivocado** (7.9 %): 4 desertores no
detectados (E004, E079, E022 y E053) y 2 falsas alarmas (E046 y E054). Los
desertores mal clasificados tienen buena asistencia (E079: 94 %, E053: 90 %) o buenas
notas (E022: 4.9, E004: 3.9), lo que sugiere causas que no están en el archivo.

**3. Dos registros nuevos.**

| Registro | Esperado | P(deserción) | Predice |
|---|---|---|---|
| Asistencia 50 %, notas 2.0 | 1 | 99.66 % | 1 |
| Asistencia 90 %, notas 4.0 | 0 | 0.56 % | 0 |

En ambos casos el resultado coincide con lo esperado.

**4. Recomendación a la entidad.** Hacer seguimiento temprano a los estudiantes con
asistencia menor a 60 % o notas menores a 2.5. En los datos limpios, el **80 %** de
los estudiantes con asistencia menor a 60 % desertó (20 de 25) y el **75 %** de los
que tienen notas menores a 2.5 también (18 de 24). Los 7 estudiantes con ambas
condiciones desertaron todos, mientras que ninguno de los 11 con asistencia de
80 % o más y notas de 3.5 o más desertó. La asistencia se puede vigilar cada
semana, antes de que lleguen las notas.

**5. Limitaciones.**
- Solo hay 76 registros y la exactitud (92.11 %) se midió con los mismos datos del
  entrenamiento, sin separar un conjunto de prueba; es una cifra optimista.
- Solo usa dos variables. Los 4 desertores no detectados (12.1 % de los 33
  desertores) tienen buenas cifras, así que faltan factores como los económicos.
- La frontera es una línea recta con umbral 0.5, que trata igual no detectar un
  desertor que dar una falsa alarma; para una entidad, lo primero cuesta más.
- Se eliminaron 8 de 84 filas (9.5 %), lo que podría sesgar el resultado.
- Es una asociación, no una causa: la baja asistencia puede ser un síntoma de la
  deserción y no su motivo.