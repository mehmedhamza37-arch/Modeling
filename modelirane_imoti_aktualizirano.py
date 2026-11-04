"""Проверимо моделиране на ИЦЖ на НСИ от реални тримесечни данни.

Вход (CSV): quarter,hpi_total,hpi_new,hpi_existing, една последователна серия
Без --input: търси HPI_2016Q1_2025Q4.csv в папката на скрипта.
Пример: python modelirane_imoti_aktualizirano.py
         python modelirane_imoti_aktualizirano.py --input /път/HPI_2016Q1_2025Q4.csv

Изход: model_comparison.csv, backtest_forecasts.csv, forecast.csv,
       ljung_box.csv, fitted_model.txt, forecast.png.

Внимание: бъдещите прогнози са условни спрямо последната дата във ВХОДНИЯ файл;
при използвания вход тук това е 2025Q4 (предварителни данни).
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox


def load_data(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Няма входен CSV файл: {source}')
    if source.suffix.lower() != '.csv':
        raise ValueError('Първо преобразувайте Excel в CSV с колони quarter,hpi_total,hpi_new,hpi_existing.')
    df = pd.read_csv(source, encoding='utf-8-sig')
    cols = ['quarter', 'hpi_total', 'hpi_new', 'hpi_existing']
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'Липсващи колони: {missing}')
    if len(df) == 0:
        raise ValueError('Входната таблица е празна.')
    df['quarter'] = pd.PeriodIndex(df['quarter'].astype(str).str.strip(), freq='Q')
    if df['quarter'].duplicated().any():
        dup = df.loc[df['quarter'].duplicated(keep=False), 'quarter'].astype(str).unique()
        raise ValueError(f'Дублирани тримесечия: {list(dup)}')
    df = df.sort_values('quarter')
    expected = pd.period_range(df['quarter'].iloc[0], df['quarter'].iloc[-1], freq='Q')
    if len(expected) != len(df):
        missing_quarters = sorted(set(expected) - set(df['quarter']))
        raise ValueError(f'Липсват тримесечия: {[str(q) for q in missing_quarters]}')
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
            raise ValueError(f'Колоната {col} има непълни, нечислови или неположителни стойности.')
    return df.set_index('quarter')[cols[1:]]


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y, float)-np.asarray(f, float))**2)))


def mape(y, f):
    y = np.asarray(y, float); f = np.asarray(f, float)
    return float(np.mean(np.abs((y-f)/y))*100)


def forecast_trend(y, h):
    x = np.arange(1, len(y)+1, dtype=float)
    model = sm.OLS(np.asarray(y, float), sm.add_constant(x)).fit()
    xf = np.arange(len(y)+1, len(y)+h+1, dtype=float)
    return np.asarray(model.predict(np.column_stack([np.ones(h), xf]))), model


def select_arima(y, max_pq=2):
    """Малък набор ARIMA(p,1,q), модел се избира по BIC, само ако е конвергирал.

    За контрол на големината/неустойчивостта на извадката: p,q<=2,
    стационарност на AR частта и обратимост на MA частта (за диференцираната серия).
    """
    candidates = []
    for p in range(max_pq+1):
        for q in range(max_pq+1):
            for tr in ('n','c'):  # без/с дрейф
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        fit = SARIMAX(y, order=(p, 1, q), trend=tr,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True).fit(disp=False, maxiter=120)
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue
                if (fit.mle_retvals.get('converged', False) and
                        np.isfinite(fit.bic) and np.isfinite(fit.aic)):
                    candidates.append((float(fit.bic), float(fit.aic), p+q, p, q, tr, fit))
    if not candidates:
        raise RuntimeError('Нито един ARIMA модел не конвергира. Проверете данните и извадката.')
    bic, aic, _, p, q, tr, fit = min(candidates, key=lambda x: (x[0],x[1],x[2]))
    return {'order': (p,1,q), 'trend':tr, 'bic':bic, 'aic':aic, 'fit':fit}


def backtest(y, start=28):
    if len(y) <= max(start, 12):
        raise ValueError('Недостатъчно наблюдения за външноизвадкова оценка.')
    observations = []
    for i in range(start, len(y)):
        train = y.iloc[:i]
        predictions = {'Наивен':float(train.iloc[-1])}
        if len(train) >= 4:
            predictions['Сезонно наивен'] = float(train.iloc[-4])
        else:
            predictions['Сезонно наивен'] = float('nan')
        predictions['Линеен тренд'] = float(forecast_trend(train,1)[0][0])
        try:
            arima = select_arima(train)
            predictions['ARIMA'] = float(arima['fit'].forecast(1).iloc[0])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            predictions['ARIMA'] = float('nan')
        for name, prediction in predictions.items():
            observations.append({'quarter':str(y.index[i]), 'model':name,
                                 'actual':float(y.iloc[i]), 'forecast':prediction})
    details = pd.DataFrame(observations)
    rows = []
    for name, group in details.groupby('model',sort=False):
        sub = group.dropna(subset=['forecast'])
        if len(sub)==0: continue
        rows.append({'model':name,'n':len(sub),'MAE':mae(sub.actual, sub.forecast),
                     'RMSE':rmse(sub.actual,sub.forecast),'MAPE':mape(sub.actual,sub.forecast)})
    comparison = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    if comparison.n.nunique()!=1:
        raise RuntimeError('Моделите имат различен брой тестови прогнози: не са пряко съпоставими.')
    return comparison, details


def main():
    default_path = Path(__file__).with_name('HPI_2016Q1_2025Q4.csv')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default=str(default_path), help='CSV с официални тримесечни данни')
    ap.add_argument('--output', default='results_hpi', help='Папка за резултати')
    ap.add_argument('--test-start',type=int,default=28,help='Брой тримесечия за първоначално обучение')
    ap.add_argument('--horizon',type=int,default=4,help='Хоризонт за екстраполация от последния входен период')
    args=ap.parse_args()
    df=load_data(args.input)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    y=df['hpi_total'].astype(float)
    scores,details=backtest(y, start=args.test_start)
    scores.to_csv(output/'model_comparison.csv',index=False,float_format='%.6f')
    details.to_csv(output/'backtest_forecasts.csv',index=False,float_format='%.6f')
    chosen=select_arima(y)
    fit=chosen['fit']
    pred=fit.get_forecast(steps=args.horizon)
    intervals=pred.conf_int(alpha=0.05)
    forecast_quarters=pd.period_range(y.index[-1]+1,periods=args.horizon,freq='Q')
    fc=pd.DataFrame({'quarter':[str(q) for q in forecast_quarters],
                     'forecast':np.asarray(pred.predicted_mean),
                     'lower95':np.asarray(intervals.iloc[:,0]),
                     'upper95':np.asarray(intervals.iloc[:,1])})
    fc.to_csv(output/'forecast.csv',index=False,float_format='%.6f')
    # Пропускаме периода за инициализация на Кaлмановия филтър и 
    # отчитаме оценените AR/MA параметри в степените на свобода.
    diagnostic_resid = fit.resid.iloc[max(5, int(fit.loglikelihood_burn)):]
    lb=acorr_ljungbox(diagnostic_resid,lags=[4],model_df=sum(chosen['order'][::2]),return_df=True)
    lb.to_csv(output/'ljung_box.csv')
    with open(output/'fitted_model.txt','w',encoding='utf-8') as f:
        f.write(f"Source: {args.input}\nObservations: {len(y)}\nLast actual: {y.index[-1]}\n")
        f.write(f"Chosen ARIMA: {chosen['order']} with trend={chosen['trend']}\n")
        f.write(f"BIC={chosen['bic']:.6f}; AIC={chosen['aic']:.6f}\n")
        f.write(f"converged={fit.mle_retvals.get('converged')}\n")
        f.write(f"Backtest start: {y.index[args.test_start]}; one step; {len(y)-args.test_start} origins\n")
        f.write(f"Ljung-Box lag4 p={float(lb.iloc[0].lb_pvalue):.6f}\n")
        f.write('\n'+fit.summary().as_text())
    xs=np.arange(len(y))
    future_xs=np.arange(len(y),len(y)+args.horizon)
    fig,ax=plt.subplots(figsize=(10.2,5.1))
    ax.plot(xs,y.values,label='Действителен ИЦЖ (НСИ)',linewidth=2)
    ax.plot(future_xs,fc.forecast,label=f"ARIMA{chosen['order']} моделна прогноза",linewidth=2,marker='o')
    ax.fill_between(future_xs,fc.lower95,fc.upper95,alpha=.20,label='95% моделни интервали')
    ax.axvline(len(y)-.5,color='gray',linestyle=':',linewidth=1)
    ticks=list(range(0,len(y)+args.horizon,4))
    if ticks[-1] != len(y)+args.horizon-1: ticks.append(len(y)+args.horizon-1)
    all_labels=[str(x) for x in y.index]+[str(x) for x in forecast_quarters]
    ax.set_xticks(ticks,labels=[all_labels[i] for i in ticks],rotation=50,ha='right')
    ax.set_ylabel('ИЦЖ (2015 = 100)')
    ax.set_xlabel('Тримесечие')
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper left',fontsize=9)
    fig.tight_layout()
    fig.savefig(output/'forecast.png',dpi=190)
    plt.close(fig)
    print(f'Източник: {args.input}; наблюдения: {len(y)} ({y.index[0]}–{y.index[-1]})')
    print(f'ARIMA{chosen["order"]}, trend={chosen["trend"]}, BIC={chosen["bic"]:.3f}')
    print(scores.to_string(index=False))
    print(fc.to_string(index=False))
    print('Резултати:',output.resolve())


if __name__=='__main__':
    main()
