import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def cargar_datos_NEA(directorio_raiz):
    data_list = []
    
    for dirpath, _, filenames in os.walk(directorio_raiz):
        for filename in filenames:
            if filename.endswith(".json") and not filename.endswith("checkpoint.json"):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    content = json.load(f)
                                
                for item in content:
                    entry = {
                        'pl_name':   item.get('pl_name'),
                        'st_host':   item.get('hostname'),
                        'st_Teff':   item.get('st_teff'),
                        'st_rad':    item.get('st_rad'),
                        'st_mass':   item.get('st_mass'),
                        'ra':        item.get('ra'),
                        'dec':       item.get('dec'),
                        'period_day':item.get('pl_orbper'),
                        'a_au':      item.get('pl_orbsmax'),
                        'a_au_err1': item.get('pl_orbsmaxerr1'),
                        'a_au_err2': item.get('pl_orbsmaxerr2'),
                        'pl_rad_e:': item.get('pl_rade'),
                        'pl_rad_e_err1': item.get('pl_radeerr1'),
                        'pl_rad_e_err2': item.get('pl_radeerr2'),
                        'pl_mass':   item.get('pl_bmasse'),
                        'ins_flux':  item.get('pl_insol'),
                        'pl_eq_temp':item.get('pl_eqt')
                    }
                    data_list.append(entry)

    df = pd.DataFrame(data_list)
    return df




def cargar_datos(directorio_raiz):
    '''
    Dado un directorio donde hay muchos subdirectorios con jsons,
    obtiene los jsons y los convierte en un data frame
    '''
    
    data_list = []
    for dirpath, _, filenames in os.walk(directorio_raiz):
        for filename in filenames:
            if filename.endswith(".json") and not filename.endswith("checkpoint.json"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = json.load(f)                        
                        entry = {
                            'target': content.get('target'),
                            'planet_name': content.get('planet_name'),
                            'period': content.get('bls', {}).get('period_days'),
                            'r_star': content.get('stellar', {}).get('radius_Rsun'),
                            'm_star': content.get('stellar', {}).get('mass_Msun'), 
                            'radius': content.get('physical', {}).get('planet_radius_Rearth'),
                            'a': content.get('physical', {}).get('semi_major_axis_AU'),
                            'from': content.get('from'),
                        }
                        
                        nea_info = content.get('NEA', {})
                        if isinstance(nea_info, dict):
                            entry.update(nea_info)
                        data_list.append(entry)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error procesando {filepath}: {e}")
    return pd.DataFrame(data_list)




R_EARTH_EQ_KM = 6378.137  # Radio Ecuatorial Terrestre
AU_KM = 149597870.7       # Unidad Astronómica

# Información de:
# https://web.archive.org/web/20250804031446/https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html
# https://web.archive.org/web/20250802141251/https://nssdc.gsfc.nasa.gov/planetary/factsheet/

# es de wayback machine porque la página estaba caída cuando intenté entrar a ella el 21/12/2025

# el ins_flux lo obtengo como 1/smax^2, ya que la luminosidad estelar es 1.
# Fórmulas: https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html

SOLAR_SYSTEM_DATA = {
    "Mercurio": {"radius": 0.383, "smax": 0.387, "period": 88.0, "teq": 440, "mass": 0.0553, "ins_flux": 6.67},
    "Venus":   {"radius": 0.949, "smax": 0.723, "period": 224.7, "teq": 737, "mass": 0.815, "ins_flux": 1.91},
    "Tierra":   {"radius": 1.0000, "smax": 1.0000, "period": 365.2, "teq": 288, "mass": 1.000, "ins_flux": 1.00},
    "Marte":    {"radius": 0.532, "smax": 1.52, "period": 687.0, "teq": 208, "mass": 0.107, "ins_flux": 0.43},
    "Jupiter": {"radius": 11.21, "smax": 5.20, "period": 4331.0, "teq": 163, "mass": 317.8, "ins_flux": 0.037},
    "Saturno":  {"radius": 9.45, "smax": 9.57, "period": 10747.0, "teq": 133, "mass": 95.2, "ins_flux": 0.011},
    "Urano":  {"radius": 4.01, "smax": 19.17, "period": 30895.0, "teq": 78, "mass": 14.5, "ins_flux": 0.0027},
    "Neptuno": {"radius": 3.88, "smax": 30.18, "period": 598000.0, "teq": 73, "mass": 17.1, "ins_flux": 0.0011}
}


def get_planet_data(planet_name: str, output_folder: str):

    data = SOLAR_SYSTEM_DATA[planet_name]

    planet_json = {
        "target": "Sun",
        "planet_name": planet_name,
        "from": "NASA",
        "stellar": {
            "radius_Rsun": 1.0,
            "mass_Msun": 1.0
        },
        "bls": {
            "period_days": data["period"]
        },
        "physical": {
            "planet_radius_Rearth": data["radius"],
            "semi_major_axis_AU": data["smax"]
        },     
        #No son de NEA, sino de NASA, pero así no hay complicaciones con el data frame
        "NEA": {
            "NEA_pl_name": planet_name,
            "NEA_st_host": "Sun",
            "NEA_st_Teff": 5772,
            "NEA_st_rad": 1.0,
            "NEA_st_mass": 1.0,
            "NEA_ra": 0.0,  # El Sol no tiene RA/Dec fijas desde la Tierra
            "NEA_dec": 0.0,
            "NEA_period_day": data["period"],
            "NEA_a_au": data["smax"],
            "NEA_pl_rad_e": data["radius"],
            "NEA_pl_mass": data["mass"],
            "NEA_ins_flux": data["ins_flux"],
            "NEA_pl_eq_temp": data["teq"]
        }
    }

    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{planet_name}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(planet_json, f, indent=4)
    
    return planet_json



def plot_y_vs_x(df, y_num, x_num, name_col="planet_name",
                y_label=None, x_label=None, lines = None, bckg_zones = None, hz_data=None,
                y_log=True, x_log=True, color_nasa='olive', color_other='magenta', 
                alpha=0.6, s=50, edgecolor='k', grid=False, save_path=None, 
                highlight_names=None, y_lim_d=None, y_lim_u=None, x_lim_d =None, x_lim_u = None, loc = 'best', inv_xaxis=False, leg_ref=None):
    
    if highlight_names is None: highlight_names = {}
    target_names = list(highlight_names.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))


    plt.rcParams.update({
        "text.usetex": False,            
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Serif", "serif"],
        "mathtext.fontset": "cm",        
        "axes.labelweight": "normal"
    })        

    
    if bckg_zones:
        x_min_global = min(z[0] for z in bckg_zones)
        x_max_global = max(z[1] for z in bckg_zones)
        ax.set_xlim(x_min_global, x_max_global)
        
        for x_min, x_max, color, label in bckg_zones:
            ax.axvspan(x_min, x_max, color=color, alpha=0.4, zorder=0, lw=0)
            ax.text(np.sqrt(x_min * min(x_max, ax.get_xlim()[1] if ax.get_xlim()[1]>0 else x_max)), 
                    0.25, 
                    label, color='#4A148C', ha='center', va='top', 
                    fontsize=12, fontweight='bold', zorder=1, rotation=0)
            
            if x_max < 1e4:
                ax.axvline(x_max, color='#4A148C', linestyle=':', lw= 0.6, alpha=0.3, zorder=1)
                ax.text(x_max-0.05, 0.04, rf"{x_max:.2f} $M_\oplus$",
                transform=ax.get_xaxis_transform(),
                rotation=90, ha='right', va='bottom', 
                color='#b06fef', fontsize=12, fontweight='bold', zorder=2)
    if hz_data:
        t_eff, f_in, f_out = hz_data
        ax.fill_betweenx(
            t_eff,     
            f_in,      
            f_out,     
            color='yellowgreen', 
            alpha=0.15,
            zorder=0,  
            label='Zona Habitable (ZH)'
        )
   
    calc_x = lambda d: d[x_num] 
    calc_y = lambda d: d[y_num]

    
    mask_nasa = df['from'] == 'NASA'
    other_data = df[~mask_nasa]

    
    ax.scatter(calc_x(other_data), calc_y(other_data) , 
               alpha=alpha, s=s, c=color_other, edgecolors='none', 
               marker='o', label='Exoplanetas (Tránsito TESS)')

    high_data = df[df[name_col].isin(target_names)]
    if not high_data.empty:
        ax.scatter(calc_x(high_data), calc_y(high_data), 
                   alpha=1, s=s, c=color_other, edgecolors='k', 
                   marker='o', zorder=10)

    nasa_data = df[mask_nasa].copy()
    nasa_data['tmp_x'] = calc_x(nasa_data)
    nasa_data['tmp_y'] = calc_y(nasa_data)
    nasa_data = nasa_data.sort_values('tmp_x')
    
    ax.scatter(nasa_data['tmp_x'], nasa_data['tmp_y'],
               alpha=1, s=s,  c=color_nasa, edgecolors=edgecolor, 
               marker='o', label='Planetas Sistema Solar', zorder=11)

    solar_map = {
        'Mercurio': 'Mer', 'Venus': 'Ven', 'Tierra': 'Tierra', 'Marte': 'Mar',
        'Jupiter': 'Júp', 'Saturno': 'Sat', 'Urano': 'Ura', 'Neptuno': 'Nep'
    }

    for i, (_, row) in enumerate(nasa_data.iterrows()):
        label = solar_map.get(row[name_col], row[name_col][:3])
        direct = 1 if i%2 == 0 else -1
        if label == 'Mer': direct = -1
        if label == 'Sat': direct = -1
        direct = 1 #Comentar para primera gráfica MR
        v_offset = 10 * direct
        v_align = 'bottom' if direct > 0 else 'top'
        
        ax.annotate(label, xy=(row['tmp_x'], row['tmp_y']), xytext=(0, v_offset),
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    ha='center', va=v_align, color = color_nasa,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.15, boxstyle='round,pad=0.2'), zorder = 10)

    if lines:
        for line in lines: 
            ax.plot(line['x'], line['y'],
                    linestyle = line.get('linestyle', '--'),
                    color = line.get('color', 'gray'),
                    linewidth = line.get('lw', 2),
                    label = line.get('label', None),
                    zorder = 5
                   )
            

    for i, (name, custom_label) in enumerate(highlight_names.items()):
        row = high_data[high_data[name_col] == name]
        direct = 1 if i%2 == 0 else -1
        v_offset = 30 * direct
        x_offset = -40
        # Comentar esto para la gráfica Zona Habitable
        # if name == 'TOI-4010 d':
        #     v_offset = -30
        #     x_offset = -20
        # if name == 'TOI-4010 b':
        #     v_offset = 15
        #     x_offset = -70
        # if name == 'TOI-715 b':
        #     v_offset = 30
        #     x_offset = -10
        if not row.empty:
            x_p, y_p = calc_x(row).iloc[0], calc_y(row).iloc[0]
            ax.annotate(custom_label, xy=(x_p, y_p),
                        xytext=(x_offset, v_offset), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color='black', lw=1),
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'), zorder = 10)

    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')

    from matplotlib.lines import Line2D

    ax = plt.gca()
    
    if leg_ref:
        leg_refs = ax.legend(
            handles=leg_ref,
            loc='center left',
            frameon=False
        )
        
        ax.add_artist(leg_refs)

    
    ax.set_xlabel(x_label or "x", fontsize = 14)
    ax.set_ylabel(y_label or "y", fontsize = 14)

    if y_lim_d is not None and y_lim_u is not None:
        ax.set_ylim(y_lim_d, y_lim_u)

    if x_lim_d is not None and x_lim_u is not None:
        ax.set_xlim(x_lim_d, x_lim_u)

    if inv_xaxis:
        plt.gca().invert_xaxis()

    ax.grid(visible=grid)
    
    
    plt.rcParams.update({
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})
    ax.legend(loc=loc, frameon=True, fontsize = 11)
    plt.tight_layout()

    if save_path: plt.savefig(save_path, dpi =300)
    plt.show()















