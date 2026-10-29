"""Miniproyecto 1: la serie del grupo, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. El
informe, docs/informe.md, cita las cifras y las figuras que produce este
guion.

La serie: ventas mensuales de las licorerías de Estados Unidos (cerveza, vino
y licor, NAICS 4453), en millones de dólares, sin ajuste estacional. Fuente:
U.S. Census Bureau, Monthly Retail Trade, tomada de FRED (MRTSSM4453USN).
"""
import itertools
import warnings

import matplotlib

matplotlib.use("Agg")
import holidays  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)

warnings.filterwarnings("ignore")
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ============================================================
# 1. La serie
# ============================================================
ARCHIVO = "datos/ventas_licores_eeuu.csv"
FECHA = "observation_date"
VALOR = "MRTSSM4453USN"
PERIODO = 12
N_PRUEBA = 24       # los meses del backtest común; preparar no los toca
ESCALA = 1000       # los modelos se ajustan en miles de millones de dólares
UMBRAL = 3          # un mes es atípico si su residuo STL pasa 3 desviaciones
CHOQUE = ("2020-03", "2021-12")   # el tramo de la pandemia donde se buscan

# Lo que se hizo con los huecos y con los datos atípicos, en una línea. Va
# a la columna `decision` de la tabla de métricas.
DECISION = ("sin huecos (regla: interpolación lineal); meses de 2020-2021 con "
            "residuo STL > 3 desv. cambiados por tendencia + estacional")


def atipicos(serie):
    """Meses del CHOQUE cuyo residuo STL robusto pasa UMBRAL desviaciones."""
    stl = STL(serie, period=PERIODO, robust=True).fit()
    z = stl.resid / stl.resid.std()
    todos = serie.index[z.abs() > UMBRAL]
    print(f"  meses con |residuo STL| > {UMBRAL} desv. en toda la serie:",
          [f"{m:%Y-%m}" for m in todos])
    en_choque = (serie.index >= CHOQUE[0]) & (serie.index <= CHOQUE[1])
    meses = serie.index[(z.abs() > UMBRAL) & en_choque]
    return meses, stl


def preparar(serie):
    """Interpola los huecos y cambia cada atípico por tendencia + estacional.

    La serie conserva el salto de nivel de 2020 (la tendencia lo recoge); lo
    que se quita es el pico de cada mes atípico. Los últimos N_PRUEBA meses,
    los del backtest, quedan como vienen.
    """
    serie = serie.interpolate(method="linear")
    meses, stl = atipicos(serie)
    limpia = serie.copy()
    limpia[meses] = (stl.trend + stl.seasonal)[meses]
    print(f"  atípicos que se cambian ({CHOQUE[0]} a {CHOQUE[1]}): {len(meses)}")
    for mes in meses:
        print(f"    {mes:%Y-%m}  observado {serie[mes]:,.0f}  ->  {limpia[mes]:,.0f}")
    return limpia


def graficar_serie(serie, cruda):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(cruda.index, cruda.to_numpy(), color="tab:red", linewidth=0.8,
             label="observada")
    eje.plot(serie.index, serie.to_numpy(), color="tab:blue", label="preparada")
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--",
                label="inicio de la prueba")
    eje.set_title("Ventas mensuales de licorerías en EE. UU. (NAICS 4453)")
    eje.set_ylabel("millones de dólares")
    eje.legend(frameon=False, fontsize=9)
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. Descomposición
# ============================================================

def descomponer(serie):
    """STL robusta con periodo 12; guarda figuras/stl.png e imprime su lectura."""
    stl = STL(serie, period=PERIODO, robust=True).fit()
    figura = stl.plot()
    figura.set_size_inches(9, 7)
    figura.tight_layout()
    figura.savefig(FIGURAS / "stl.png", dpi=120)
    plt.close(figura)

    tendencia = stl.trend
    print(f"  tendencia: {tendencia.iloc[0]:,.0f} en {serie.index[0]:%Y-%m} y "
          f"{tendencia.iloc[-1]:,.0f} en {serie.index[-1]:%Y-%m} "
          f"({tendencia.iloc[-1] / tendencia.iloc[0]:.1f} veces)")
    for anio in (2019, 2021):
        print(f"  tendencia en diciembre de {anio}: {tendencia[f'{anio}-12'].iloc[0]:,.0f}")
    por_mes = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    print("  estacional promedio por mes:",
          {mes: round(valor) for mes, valor in por_mes.items()})
    print(f"  mes más alto {por_mes.idxmax()} ({por_mes.max():+,.0f}), "
          f"mes más bajo {por_mes.idxmin()} ({por_mes.min():+,.0f})")
    diciembre = stl.seasonal[stl.seasonal.index.month == 12]
    print(f"  estacional de diciembre: {diciembre.iloc[0]:+,.0f} en "
          f"{diciembre.index[0]:%Y} y {diciembre.iloc[-1]:+,.0f} en {diciembre.index[-1]:%Y}")
    resid = stl.resid
    grande = resid.abs().idxmax()
    print(f"  residuo más grande: {grande:%Y-%m} ({resid[grande]:+,.0f}); "
          f"desviación estándar del residuo {resid.std():,.0f}")
    return stl


# ============================================================
# 3. Estacionariedad e identificación
# ============================================================

def diferenciar(serie, d, D):
    x = serie
    if D:
        x = x.diff(PERIODO)
    if d:
        x = x.diff()
    return x.dropna()


def fuera(valores, intervalo):
    """Rezagos (sin el 0) cuyo valor sale de la banda centrada en cero."""
    banda = intervalo[:, 1] - valores
    return [k for k in range(1, len(valores)) if abs(valores[k]) > banda[k]]


def estacionariedad(serie):
    """ADF sobre las cuatro versiones, elección de d y D, y ACF y PACF de la elegida."""
    versiones = [("original", 0, 0, "ct"), ("diff()", 1, 0, "c"),
                 ("diff(12)", 0, 1, "c"), ("diff(12).diff()", 1, 1, "c")]
    filas = []
    for nombre, d, D, tipo in versiones:
        x = diferenciar(serie, d, D)
        estadistico, p, rezagos, *_ = adfuller(x, regression=tipo, autolag="AIC")
        filas.append({"version": nombre, "d": d, "D": D, "regression": tipo,
                      "estadistico": round(estadistico, 3), "rezagos": rezagos,
                      "valor p": round(p, 4), "p < 0,05": p < 0.05})
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    _, p_c, *_ = adfuller(serie, regression="c", autolag="AIC")
    print(f"  original con regression='c': valor p {p_c:.4f}")

    # ADF solo mira la raíz unitaria del mes anterior. El patrón anual que
    # queda se mira en el rezago 12 de la ACF: si sigue cerca de 1, falta D.
    tabla["ACF r12"] = [round(acf(diferenciar(serie, d, D), nlags=12)[12], 2)
                        for _, d, D, _ in versiones]
    print(tabla[["version", "valor p", "ACF r12"]].to_string(index=False))
    # la combinación con menos diferencias que pasa ADF y deja |r12| < 0,5
    pasan = tabla[tabla["p < 0,05"] & (tabla["ACF r12"].abs() < 0.5)]
    pasan = pasan.assign(total=lambda t: t["d"] + t["D"])
    elegida = pasan.sort_values(["total", "D"], ascending=[True, False]).iloc[0]
    d, D = int(elegida["d"]), int(elegida["D"])
    print(f"  elegido: d = {d}, D = {D} ({elegida['version']})")

    x = diferenciar(serie, d, D)
    print(f"  banda de confianza: ±{1.96 / np.sqrt(len(x)):.3f} ({len(x)} meses)")
    print("  ACF fuera: ", fuera(*acf(x, nlags=36, alpha=0.05)))
    print("  PACF fuera:", fuera(*pacf(x, nlags=36, alpha=0.05, method="ywm")))
    r = acf(x, nlags=24)
    rp = pacf(x, nlags=24, method="ywm")
    print(f"  ACF:  r1 {r[1]:5.2f}  r2 {r[2]:5.2f}  r12 {r[12]:5.2f}  r24 {r[24]:5.2f}")
    print(f"  PACF: r1 {rp[1]:5.2f}  r2 {rp[2]:5.2f}  r12 {rp[12]:5.2f}  r24 {rp[24]:5.2f}")

    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    plot_acf(x, lags=36, ax=ejes[0], title=f"ACF, d = {d}, D = {D}")
    plot_pacf(x, lags=36, ax=ejes[1], title=f"PACF, d = {d}, D = {D}", method="ywm")
    for eje in ejes:
        eje.set_ylim(-1.05, 1.05)
        eje.set_xticks(range(0, 37, 6))
        eje.set_xlabel("rezago (meses)")
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120)
    plt.close(figura)
    return d, D


# ============================================================
# 4. Modelos clásicos
# ============================================================
# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

def elegir_sarima(serie, d, D):
    """Rejilla de 16 órdenes (p, q, P, Q) en {0, 1} comparada por AIC y Ljung-Box."""
    escalada = serie / ESCALA
    filas = []
    for p, q, P, Q in itertools.product([0, 1], repeat=4):
        ajuste = ARIMA(escalada, order=(p, d, q),
                       seasonal_order=(P, D, Q, PERIODO)).fit()
        filas.append({"order": (p, d, q), "seasonal_order": (P, D, Q, PERIODO),
                      "AIC": round(ajuste.aic, 1)})
    rejilla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    print(rejilla.to_string())
    orden = rejilla.loc[0, "order"]
    estacional = rejilla.loc[0, "seasonal_order"]

    mejor = ARIMA(escalada, order=orden, seasonal_order=estacional).fit()
    print(f"  elegido: SARIMA{orden}{estacional}, AIC {mejor.aic:.1f}, "
          f"{rejilla.loc[1, 'AIC'] - rejilla.loc[0, 'AIC']:.1f} por debajo del segundo")
    print("  coeficientes:", mejor.params.round(3).to_dict())
    print("  valores p:   ", mejor.pvalues.round(3).to_dict())

    resid = mejor.resid.iloc[d + PERIODO * D:]   # los primeros son de arranque
    print(f"  residuos que quedan: {len(resid)}; desviación {resid.std() * ESCALA:,.0f}")
    print(acorr_ljungbox(resid, lags=[12, 24]).round(3).to_string())
    fuera_resid = fuera(*acf(resid, nlags=24, alpha=0.05))
    print("  ACF de los residuos fuera de la banda:", fuera_resid)
    for mes in resid.abs().sort_values(ascending=False).index[:3]:
        print(f"    residuo grande {mes:%Y-%m}: {resid[mes] * ESCALA:+,.0f}")

    figura, (arriba, abajo) = plt.subplots(2, 1, figsize=(9, 5))
    arriba.bar(resid.index, resid * ESCALA, width=20, color="tab:blue")
    arriba.axhline(0, color="black", linewidth=0.8)
    arriba.set_ylabel("millones de dólares")
    arriba.set_title(f"Residuos de SARIMA{orden}{estacional}")
    plot_acf(resid, lags=24, ax=abajo, title="")
    abajo.set_ylim(-1.05, 1.05)
    abajo.set_xticks(range(0, 25, 3))
    abajo.set_xlabel("rezago (meses)")
    abajo.set_ylabel("ACF")
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120)
    plt.close(figura)
    return orden, estacional


def modelo_sarima(orden, orden_estacional):
    """El SARIMA elegido, ajustado en cada origen sobre `entrenamiento / ESCALA`."""
    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, order=orden,
                       seasonal_order=orden_estacional).fit()
        return ajuste.forecast(h).to_numpy() * ESCALA
    return modelo


def exogenas(serie):
    """Viernes y sábados, y festivos federales de EE. UU., de cada mes.

    Las dos salen del calendario, así que se conocen años antes. Cubren la
    serie y los 12 meses que siguen a su último dato.
    """
    indice = pd.date_range(serie.index[0], periods=len(serie) + 12, freq="MS")
    dias = pd.date_range(indice[0], indice[-1] + pd.offsets.MonthEnd(1), freq="D")
    fines = pd.Series(dias.dayofweek.isin([4, 5]), index=dias).resample("MS").sum()
    calendario = holidays.country_holidays("US", years=range(indice[0].year,
                                                             indice[-1].year + 1))
    festivos = pd.Series(1, index=pd.to_datetime(list(calendario.keys())))
    festivos = festivos.resample("MS").sum()
    x = pd.DataFrame({"viernes_sabados": fines.reindex(indice, fill_value=0),
                      "festivos": festivos.reindex(indice, fill_value=0)},
                     index=indice).astype(float)
    print("  exógenas, promedio:", x.loc[serie.index].mean().round(2).to_dict(),
          "| viernes y sábados entre", int(x["viernes_sabados"].min()), "y",
          int(x["viernes_sabados"].max()))
    return x


def modelo_sarimax(orden, orden_estacional, x):
    """El SARIMA elegido con las exógenas x; solo ve las del entrenamiento y las del calendario futuro."""
    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, exog=x.loc[entrenamiento.index],
                       order=orden, seasonal_order=orden_estacional).fit()
        futuro = fechas_siguientes(entrenamiento, h)
        return ajuste.forecast(h, exog=x.loc[futuro]).to_numpy() * ESCALA
    return modelo


def revisar_sarimax(serie, orden, estacional, x):
    """Coeficientes de las exógenas con toda la serie y AIC con y sin ellas."""
    escalada = serie / ESCALA
    sin = ARIMA(escalada, order=orden, seasonal_order=estacional).fit()
    con = ARIMA(escalada, exog=x.loc[serie.index], order=orden,
                seasonal_order=estacional).fit()
    for nombre in x.columns:
        print(f"  {nombre}: coeficiente {con.params[nombre] * ESCALA:+,.1f} millones "
              f"por unidad, valor p {con.pvalues[nombre]:.3f}")
    print(f"  AIC sin exógenas {sin.aic:.1f}, con exógenas {con.aic:.1f}")
    arranque = (orden[1] + estacional[1] * PERIODO)
    resid = con.resid.iloc[arranque:]
    print(f"  residuos del SARIMAX: desviación {resid.std() * ESCALA:,.0f}")
    print(acorr_ljungbox(resid, lags=[12, 24]).round(3).to_string())


# ============================================================
# 5. La tabla de métricas
# ============================================================

def fila(nombre, serie, modelo):
    por_origen, promedio = backtest(serie, modelo)
    return por_origen, {"modelo": nombre, "decision": DECISION,
                        "RMSE": round(promedio["RMSE"], 2),
                        "MAE": round(promedio["MAE"], 2),
                        "MAPE": round(promedio["MAPE"], 2)}


def errores_por_mes(serie, modelo):
    """Error absoluto medio del backtest según el mes del año que se pronostica."""
    errores = []
    for corte in range(len(serie) - N_PRUEBA, len(serie) - 12 + 1):
        entrenamiento = serie.iloc[:corte]
        futuro = fechas_siguientes(entrenamiento, 12)
        for fecha, valor in zip(futuro, modelo(entrenamiento, 12)):
            errores.append({"mes": fecha, "error": serie[fecha] - valor})
    errores = pd.DataFrame(errores)
    por_mes = errores.groupby(errores["mes"].dt.month)["error"].apply(
        lambda e: e.abs().mean())
    print("  error absoluto medio por mes del año:",
          {mes: round(valor) for mes, valor in por_mes.items()})
    sesgo = errores.groupby("mes")["error"].mean().sort_values()
    print("  meses más sobreestimados:",
          {f"{m:%Y-%m}": round(v) for m, v in sesgo.head(3).items()})
    print("  meses más subestimados:  ",
          {f"{m:%Y-%m}": round(v) for m, v in sesgo.tail(3).items()})


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el primer y el último origen del backtest, contra lo observado."""
    primero = len(serie) - N_PRUEBA
    ultimo = len(serie) - 12
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for eje, corte in zip(ejes, (primero, ultimo)):
        entrenamiento = serie.iloc[:corte]
        futuro = fechas_siguientes(entrenamiento, 12)
        eje.plot(serie.iloc[corte - 24:corte + 12], color="black", label="observado")
        for nombre, modelo in modelos.items():
            eje.plot(futuro, modelo(entrenamiento, 12), marker=".", label=nombre)
        eje.set_title(f"origen {serie.index[corte - 1]:%Y-%m}")
        eje.grid(alpha=0.3)
        eje.tick_params(axis="x", labelrotation=30)
    ejes[0].set_ylabel("millones de dólares")
    ejes[1].legend(frameon=False, fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def main():
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    print("\n1. La serie")
    serie = preparar(cruda)
    graficar_serie(serie, cruda)

    print("\n2. Componentes")
    descomponer(serie)

    print("\n3. Líneas base y backtest")
    origen_ing, fila_ing = fila("ingenuo", serie, ingenuo)
    _, fila_est = fila("estacional ingenuo", serie,
                       lambda t, h: estacional_ingenuo(t, h, PERIODO))
    print(f"  orígenes: {len(origen_ing)}, del {origen_ing['origen'].iloc[0]:%Y-%m} "
          f"al {origen_ing['origen'].iloc[-1]:%Y-%m}; prueba de "
          f"{serie.index[-N_PRUEBA]:%Y-%m} a {serie.index[-1]:%Y-%m}")
    filas = [fila_ing, fila_est]

    print("\n4. Estacionariedad e identificación")
    d, D = estacionariedad(serie)

    print("\n5. Modelos clásicos")
    orden, estacional = elegir_sarima(serie, d, D)
    sarima = modelo_sarima(orden, estacional)
    filas.append(fila(f"SARIMA{orden}{estacional}", serie, sarima)[1])
    x = exogenas(serie)
    revisar_sarimax(serie, orden, estacional, x)
    sarimax = modelo_sarimax(orden, estacional, x)
    filas.append(fila(f"SARIMAX{orden}{estacional} + exógenas", serie, sarimax)[1])

    print("\n6. Tabla de métricas")
    tabla = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    mejor = tabla.iloc[0]
    base = tabla[tabla["modelo"].isin(["ingenuo", "estacional ingenuo"])].iloc[0]
    print("Mejor modelo por RMSE:", mejor["modelo"])
    print("Mejor línea base:", base["modelo"])
    for metrica in ("RMSE", "MAE", "MAPE"):
        mejora = 100 * (base[metrica] - mejor[metrica]) / base[metrica]
        print(f"  {metrica}: {base[metrica]:,.2f} -> {mejor[metrica]:,.2f} "
              f"({mejora:.1f} % menos)")
    print("Dónde falla el SARIMAX en el backtest (real - pronóstico):")
    errores_por_mes(serie, sarimax)

    graficar_pronosticos(serie, {
        "estacional ingenuo": lambda t, h: estacional_ingenuo(t, h, PERIODO),
        "SARIMA": sarima, "SARIMAX": sarimax})


if __name__ == "__main__":
    main()
