import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from lightkurve.correctors import PLDCorrector
from astropy.constants import G, R_sun, M_sun, R_earth
from astroquery.mast import Catalogs
import warnings
import json
import os
import pandas as pd
from numpy import ma 
warnings.filterwarnings("ignore")


# ________ OBTENCIÓN Y ANÁLISIS TRÁNSITO TESS _____________ 

def get_multisector_lc(target_name, method="LCF", author="SPOC", pca_components=5):
    """
    Descarga, filtra y unifica sectores.
    """
    print(f"--- Buscando datos para {target_name} [{method}] ---")
    
    if method == "LCF":
        search = lk.search_lightcurve(target_name, author=author)
    elif method =="TPF":
        search = lk.search_targetpixelfile(target_name)
        if len(search) == 0:
            search = lk.search_tesscut(target_name)

    if len(search) == 0:
        print("No se encontraron datos.")
        return None

    print(f"Sectores totales encontrados: {len(search)}")
    
    valid_lcs = []

    for i, item in enumerate(search):
        try:
            if method == "LCF":
                lc = item.download(quality_bitmask='default')
            else:
                tpf = item.download(cutout_size=15, quality_bitmask='default')


                # Si en algún momento quiero trabajar TPF general, QUITA ESTO!!!!!!!!!!!!!!!!
                # Le pongo un filtro para quitar todos aquellos con BJT menor a 2500 porque me daba señal plana errónea antes de ese momento.
                if tpf.time[0].value < 2500:
                    continue
                
                aper_mask = tpf.create_threshold_mask(threshold=10, reference_pixel='center')

                pld = PLDCorrector(tpf, aperture_mask=aper_mask)
                lc = pld.correct(pca_components=pca_components)

            lc = lc.remove_nans().normalize().remove_outliers(sigma=5)

            if len(lc) < 100:
                continue
                            
            if np.nanstd(lc.flux) < 1e-5: 
                continue

            valid_lcs.append(lc)
            
        except Exception as e:
            print(f"Error procesando sector {i}: {e}")
            continue

    if len(valid_lcs) == 0:
        print("No tenemos datos válidos de calidad")
        return None

    lc_stitched = lk.LightCurveCollection(valid_lcs).stitch().remove_nans()
    
    return lc_stitched


def clean_lc(lc, window_length=1001, sigma=5):
    ''' 
    Limpieza de lightcurve (flateneado + remove outliers)
    '''
    if lc is None: return None
    clc = lc.copy()
    if window_length % 2 == 0: window_length += 1
    clc = clc.flatten(window_length=window_length, break_tolerance=10).remove_outliers(sigma=sigma)
    return clc



def bls(lc, min_p=0.5, max_p=30, n_points=100000):
    '''
    Ejecutar box least squares
    '''
    if lc is None: return None
    
    periodogram = lc.to_periodogram(
        method='bls', 
        period=np.linspace(min_p, max_p, n_points),
        frequency_factor=1000 
    )
    
    results = {
        'period': periodogram.period_at_max_power,
        't0': periodogram.transit_time_at_max_power,
        'depth': periodogram.depth_at_max_power,
        'duration': periodogram.duration_at_max_power,
        'periodogram': periodogram 
    }
    return results

def get_stellar_params(target_name):
    '''
    Obtener parámetros estelares básicos de la estrella host     
    '''
    
    catalog_data = Catalogs.query_object(target_name, radius=0.02, catalog="TIC")
    
    star_data = catalog_data[0]
    
    r_star = star_data['rad']
    m_star = star_data['mass']
    
    r_err = star_data['e_rad']
    m_err = star_data['e_mass']
    
    if np.isnan(r_err): r_err = 0.0
    if np.isnan(m_err): m_err = 0.0

    if np.isnan(r_star) or np.isnan(m_star):
        return None

    star_data = {
        'r_star': r_star,
        'm_star': m_star,
        'r_star_err': r_err, 
        'm_star_err': m_err 
    }
    print("\n ---- Parámetros estelares obtenidos ----\n")
    return star_data




def compute_params(bls_results, star_data, lc=None, T=None):
    '''
    Cálculo parámetros planetarios a partir de datos de tránsito
    '''
    if bls_results is None: return None

    rstar = star_data['r_star']
    if np.isnan(rstar): return None
    mstar = star_data['m_star']
    if np.isnan(mstar): return None

    rstar_err = star_data.get('r_star_err', 0.0)
    mstar_err = star_data.get('m_star_err', 0.0)
    
    R_star =  rstar * R_sun
    R_star_err = rstar_err * R_sun
    M_star =  mstar * M_sun
    M_star_err = mstar_err * M_sun
    
    depth = bls_results['depth'].value

    if T is None: T = bls_results['period']
    elif not isinstance(T, u.Quantity): T = T * u.d

    if lc is None: 
        print('No has introducido la curva de luz')
        depth_err = depth * 0.1 #Estimación bruta
    else:
        sigma = np.nanstd(lc.flux.value)

        period_days = bls_results['period'].value
        duration_days = bls_results['duration'].value

        q = duration_days / period_days

        n = len(lc)

        depth_err = sigma / (np.sqrt(n*q))


    
    # 1. Radio del Planeta: Rp = R_star * sqrt(depth)
    Rp = R_star * np.sqrt(depth)
    
    err_rel_rstar = (rstar_err / rstar)**2 if rstar > 0 else 0
    err_rel_depth = (0.5 * depth_err / depth)**2 if depth > 0 else 0
    Rp_err = Rp *  np.sqrt(err_rel_rstar + err_rel_depth)
    
    
    # 2. Semi-eje mayor: (3ª Ley de Kepler): a = [(GM*T^2)/(4*pi^2)]^(1/3)
    a = ((G * M_star * (T.to(u.s))**2) / (4 * np.pi**2))**(1/3)


    err_rel_mstar = (1/3 * mstar_err / mstar)**2 if mstar > 0 else 0
    a_err = a * np.sqrt(err_rel_mstar)


    
    print("\n--- RESULTADOS FÍSICOS ---\n")
    print(f"Radio Planeta: {Rp.to(R_earth).value:.2f} R_earth")
    print(f"Distancia semi-eje mayor(a): {a.to(u.AU).value:.4f} UA")
    
    return {
        'radius': Rp.to(R_earth), 
        'radius_err': Rp_err.to(R_earth),
        'a': a.to(u.AU),
        'a_err': a_err.to(u.AU),
        'snr': depth / depth_err
    }



# _______________ visualización + guardado/exportación _______________
    
def plots(lc=None, bls_results=None, target_name="Target", T=None, save_path=None, planet_name=None, method = "LCF"):
    '''
    Genera gráficos del stitch, periodograma y tránsito faseado
    '''
    
    if lc is None: return None
    if bls_results is None: return None
    if T is None: T = bls_results['period']
    elif not isinstance(T, u.Quantity): T = T * u.d
    
    t0 = bls_results['t0']

    plt.rcParams.update({
        "text.usetex": False,            
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Serif", "serif"],
        "mathtext.fontset": "cm",        
        "axes.labelweight": "normal"
    })

    plt.rcParams.update({
        "xtick.labelsize": 13,
        "ytick.labelsize": 13
    })
    
    # Gráfica de Lightcurve stitched

    fig0, ax0 = plt.subplots(figsize=(10, 4))
    lc.scatter(ax=ax0, s=0.5, alpha=0.5, label='Flujo', color='black')
    ax0.set_xlabel("Tiempo [días]", fontsize=14)
    ax0.set_ylabel("Flujo normalizado", fontsize=14)
    ax0.legend(loc='upper left')


    fig0.tight_layout()

    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        name = f"{target_name}_{method}_{T.value:.1f}_lightcurve_full.png"
        path = os.path.join(save_path, name)
        fig0.savefig(path, dpi=300)

    # Gráfica de periodograma
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.axvline(x=T.to_value(u.d), color='fuchsia', linestyle=':', linewidth=2, alpha=0.5, label = 'Periodo de máxima potencia', zorder =10)
    bls_results['periodogram'].plot(ax=ax1, view='period', label =None)
    ax1.set_xlabel("Periodo [días]", fontsize = 14)
    ax1.set_ylabel("Potencia BLS", fontsize = 14)
    ax1.legend(loc='upper right')


    fig1.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        name = f"{target_name}_{method}_{T.value:.1f}_periodogram.png"
        path = os.path.join(save_path, name)
        fig1.savefig(path, dpi=300)

    
    # Gráfica de tránsito faseado
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    folded = lc.fold(period=T, epoch_time=t0)
    folded.scatter(ax=ax2, s=1, alpha=0.1, c='k', label = 'Flujo observado (TESS)')
    folded.bin(time_bin_size=15*u.min).plot(
    ax=ax2,
    marker='o',
    markerfacecolor='mediumvioletred', 
    markeredgecolor='none', 
    markersize=5, 
    linestyle='-',
    color='fuchsia',
    lw=2,
    label = 'Flujo medio por intervalos'
    )
    ax2.set_xlabel("Tiempo desde tránsito central [horas]", fontsize = 14)
    ax2.set_ylabel("Flujo Normalizado", fontsize = 14)
    duration_hr = float(bls_results['duration'].to(u.hr).value)
    ax2.set_xlim(-0.1*duration_hr, 0.1*duration_hr)
    ax2.legend(loc='upper left')
    fig2.tight_layout()


    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        name = f"{target_name}_{method}_{T.value:.1f}_transit.png"
        path = os.path.join(save_path, name)
        fig2.savefig(path, dpi=300)
        print(f"\n ---- Plots guardados en carpeta: {save_path} ----- \n")

    if save_path is None:
        print("\n -------- Atención: NO HAS GUARDADO LAS IMÁGNENES --------- \n")
    plt.show()

    
def pack_results(target, planet_name, bls_results, star_data, params):
    """Empaqueta todos los resultados en un diccionario"""              
    return {
        "target": target,
        "planet_name": planet_name,
        "from": "Transit (TESS)", 
        "stellar": {
            "radius_Rsun": star_data["r_star"],
            "mass_Msun": star_data["m_star"]
        },
        "bls": {
            "period_days": bls_results["period"].value,
            "t0": bls_results["t0"].value,
            "depth": bls_results["depth"].value,
            "duration_days": bls_results["duration"].value,
        },
        "physical": {
            "planet_radius_Rearth": params["radius"].value,
            "planet_radius_err_Re": params["radius_err"].value,
            "semi_major_axis_AU": params["a"].value,
            "a_err": params["a_err"].value,
            "snr": params["snr"]
        }
    }



def pack_results_NEA(target, planet_name, bls_results, star_data, params, df_NEA):
    """
    Empaqueta todos los resultados en un diccionario

    También contiene la información de NEA 
    """              

    mask = (df_NEA['st_host'] == target)   
    df_row = df_NEA[mask]
    nea_data = {f"NEA_{k}": v for k, v in df_row.iloc[0].to_dict().items()}
    
    return {
        "target": target,
        "planet_name": planet_name,
        "from": "Transit (TESS)", 
        "stellar": {
            "radius_Rsun": star_data["r_star"],
            "mass_Msun": star_data["m_star"]
        },
        "bls": {
            "period_days": bls_results["period"].value,
            "t0": bls_results["t0"].value,
            "depth": bls_results["depth"].value,
            "duration_days": bls_results["duration"].value,
        },
        "physical": {
            "planet_radius_Rearth": params["radius"].value,
            "semi_major_axis_AU": params["a"].value,
        },
        "NEA": nea_data
    }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, ma.MaskedArray): return obj.filled(np.nan).tolist()
        return super(NpEncoder, self).default(obj)

def save_json(data, save_path):
    '''
    Guardar el json con la nueva información obtenida 
    '''
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f"{data['planet_name']}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4, cls=NpEncoder)
        print(f"JSON guardado correctamente en: {path}")
    if save_path is None:
        print("\n -------- Atención: NO HAS GUARDADO LAS IMÁGNENES --------- \n")
    

def cargar_datos_small(directorio_raiz):
    '''
    Función para ver todos los datos de los planetas a la vez, para no tener que irme metiendo de uno en uno en las carpetas
    '''
    data_list = []
    i = 0
    for dirpath, _, filenames in os.walk(directorio_raiz):

        if "checkpoint" in dirpath.lower():
            continue
        
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if "checkpoint" in filename.lower():
                continue

            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r') as f:
                content = json.load(f)
            if i % 2 == 0:
                file_type = 'LCF'
            else:
                file_type = 'TPF'
                
            entry = {
                'planet_name': content.get('planet_name'),
                'radius': content.get('physical', {}).get('planet_radius_Rearth'),
                'radius_err': content.get('physical', {}).get('planet_radius_err_Re'),
                'a': content.get('physical', {}).get('semi_major_axis_AU'),
                'a_err': content.get('physical', {}).get('a_err'),
                'type' : file_type
            }
            i += 1
            data_list.append(entry)

    return pd.DataFrame(data_list)        
# Referencias:
# https://avanderburg.github.io/tutorial/tutorial2.html
# https://lightkurve.github.io/lightkurve/tutorials/index.html
# lk.show_citation_instructions()