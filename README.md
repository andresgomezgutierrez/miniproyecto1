# Miniproyecto 1 — De las líneas base al mejor modelo clásico

Pronóstico de Series Temporales con IA · Universidad Autónoma de Occidente,
2026-II · Carlos Andrés Delgado Saavedra

El grupo toma una serie mensual propia y la lleva de principio a fin por los
módulos 1 y 2: decisiones sobre los datos, descomposición, líneas base,
*backtest*, estacionariedad, SARIMA y SARIMAX. El enunciado completo, con la
rúbrica, está publicado en el campus. Este repositorio es el punto de partida
del grupo.

## El enfoque

El miniproyecto es un experimento, no una competencia por el número más
bajo. Se evalúa que el grupo use las herramientas del curso sobre su serie,
tome decisiones y explique con sus cifras qué pasó y por qué. Un modelo que
no gana, bien analizado, vale más que uno que gana sin explicación.

La serie es propia y cumple unos mínimos: mensual, pública (con fuente y
licencia), al menos 60 observaciones y estacionalidad visible. Su ficha va
en `datos/README.md`.

## Integrantes

Los grupos son de 3 integrantes.

Esta tabla identifica al grupo. Sin ella, o con algún dato incompleto, la
entrega pierde el 20 %; quien no aparezca aquí no cuenta como parte del grupo.

| Nombre completo | Código | Correo institucional |
|---|---|---|
| Wilson Andres Gomez Gutierrez | 22601217 | wilson_and.gomez@uao.edu.co |

Trabajo individual.

## Entrega

| Qué | Fecha y hora |
|---|---|
| Cierre de la entrega | sábado 31 de octubre de 2026, 7:59 a. m. |
| Sustentación | sábado 31 de octubre de 2026, en la clase de las 8:00 a. m. |

## Cómo se entrega

1. **Se trabaja sobre un fork de este repositorio.** El grupo hace el fork,
   lo deja público, agrega a los otros dos integrantes como colaboradores y
   cada uno trabaja con su propia cuenta, en ramas. La página *Git y GitHub
   en el curso* del campus explica cómo. Una entrega que no sea fork de este
   repositorio pierde el 30 % de la nota.
2. **Un solo integrante pega el enlace del fork en la tarea *Miniproyecto 1*
   del campus**, y esa entrega vale para todos los que aparecen en la tabla
   de integrantes. Si un grupo entrega más de una vez, se califica la de
   menor nota. Nada más: ni archivos, ni comprimidos, ni código por correo.
   El grupo que no registre el enlace dentro del plazo tiene 0.0.
3. **Se califica el último commit anterior a la hora de cierre.** Lo que se
   suba después no se tiene en cuenta.
4. **El informe es `docs/informe.md`, en Markdown**, con las fórmulas en
   LaTeX y las figuras que genera el guion en `figuras/`. No se aceptan PDF,
   DOCX, presentaciones ni el informe como enlace a otro sitio: lo que no
   está en `docs/informe.md` no se califica.
5. El historial de commits se revisa para ver el aporte de cada integrante,
   y ese aporte es un factor de la nota. Los cambios de puro formato o de
   comentarios no cuentan.

## Cómo está organizado

```
miniproyecto.py      el guion del grupo, en cinco partes con TODO
comun.py             carga, métricas, líneas base y backtest; no se modifica
datos/               la serie del grupo y su ficha (datos/README.md)
docs/informe.md      la plantilla del informe; se llena en el mismo archivo
figuras/             las figuras que genera el guion
resultados/          metricas.csv, la tabla que genera el guion
```

`comun.py` es igual para todos los grupos: el mismo *backtest* (origen móvil
sobre los últimos 24 meses, horizonte de 1 a 12) y las mismas métricas
(RMSE, MAE y MAPE). Si se modifica, la parte de líneas base y *backtest*
vale 0.

`miniproyecto.py` trae la serie de ejemplo y las líneas base ya puestas, y
corre desde el primer día. Las partes marcadas con `TODO` las completa el
grupo; mientras falten, el guion las reporta como pendientes y sigue.

## Cómo se corre

```bash
pip install -r requirements.txt
python miniproyecto.py
```

En la sustentación, un integrante elegido en el momento corre el guion. La
acción de GitHub del repositorio lo corre en cada *push*: si queda en rojo,
el guion no corre de principio a fin.
