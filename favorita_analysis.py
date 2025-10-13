# =============================================================================
# IMPLEMENTACIÓN COMPLETA - ANÁLISIS EXPLORATORIO CORPORACIÓN FAVORITA
# VERSIÓN CORREGIDA - Dimensiones ajustadas y guardado funcional
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro
import warnings
from typing import List, Optional, Dict, Tuple
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuración global para estilo académico y formato carta
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

class FavoritaEDA:
    """
    Clase para análisis exploratorio completo del dataset de Corporación Favorita.
    Diseñada para generar visualizaciones de calidad académica en formato carta.
    """
    
    def __init__(self, data_path: str, color_palette: str = "Set2"):
        """
        Inicializa la clase con la ruta de los datos.
        
        Parameters:
        -----------
        data_path : str
            Ruta al directorio que contiene los archivos CSV
        color_palette : str
            Paleta de colores para las visualizaciones
        """
        self.data_path = data_path
        self.color_palette = color_palette
        self.datasets = {}
        self.results = {}
        
    def load_favorita_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los datasets de Corporación Favorita.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Diccionario con todos los datasets cargados
        """
        files_to_load = {
            'train': 'train.csv',
            'test': 'test.csv',
            'stores': 'stores.csv',
            'items': 'items.csv',
            'transactions': 'transactions.csv',
            'oil': 'oil.csv',
            'holidays_events': 'holidays_events.csv'
        }
        
        print("Cargando datasets de Corporación Favorita...")
        for name, filename in files_to_load.items():
            file_path = os.path.join(self.data_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    self.datasets[name] = df
                    print(f"✓ {name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                except Exception as e:
                    print(f"✗ Error cargando {filename}: {e}")
            else:
                print(f"✗ Archivo no encontrado: {filename}")
        
        return self.datasets
    
    def comprehensive_data_overview(self, figsize: Tuple[int, int] = (8.5, 11)) -> plt.Figure:
        """
        Genera un resumen visual completo de todos los datasets.
        Formato carta: 8.5 x 11 pulgadas
        """
        if not self.datasets:
            print("No hay datasets cargados. Ejecute load_favorita_datasets() primero.")
            return None
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.3)
        
        # 1. Resumen de dimensiones - ARRIBA COMPLETO
        ax1 = fig.add_subplot(gs[0, :])
        dataset_info = pd.DataFrame({
            'Dataset': list(self.datasets.keys()),
            'Filas': [df.shape[0] for df in self.datasets.values()],
            'Columnas': [df.shape[1] for df in self.datasets.values()]
        })
        
        x = np.arange(len(dataset_info))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, dataset_info['Filas'], width, 
                       label='Filas', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, dataset_info['Columnas'], width, 
                       label='Columnas', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Datasets')
        ax1.set_ylabel('Cantidad (escala log)')
        ax1.set_title('Dimensiones de los Datasets - Corporación Favorita', fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dataset_info['Dataset'], rotation=45, ha='right')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Añadir valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{int(height):,}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # 2. Tipos de datos por dataset - IZQUIERDA
        ax2 = fig.add_subplot(gs[1, 0])
        if 'train' in self.datasets:
            train_types = self.datasets['train'].dtypes.value_counts()
            ax2.pie(train_types.values, labels=train_types.index, autopct='%1.1f%%',
                   colors=sns.color_palette(self.color_palette, len(train_types)))
            ax2.set_title('Tipos de Datos\n(Training Set)', fontweight='bold')
        
        # 3. Valores nulos por dataset - DERECHA
        ax3 = fig.add_subplot(gs[1, 1])
        null_data = []
        for name, df in self.datasets.items():
            null_count = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            null_percentage = (null_count / total_cells) * 100 if total_cells > 0 else 0
            null_data.append({'Dataset': name, 'Null_Percentage': null_percentage})
        
        null_df = pd.DataFrame(null_data)
        bars = ax3.bar(null_df['Dataset'], null_df['Null_Percentage'], 
                      color=sns.color_palette(self.color_palette, len(null_df)))
        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('% Valores Nulos')
        ax3.set_title('Porcentaje de Valores Nulos', fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Distribución temporal - IZQUIERDA
        ax4 = fig.add_subplot(gs[2, 0])
        if 'train' in self.datasets and 'date' in self.datasets['train'].columns:
            train_df = self.datasets['train'].copy()
            train_df['date'] = pd.to_datetime(train_df['date'])
            daily_sales = train_df.groupby('date')['unit_sales'].sum().rolling(7).mean()
            ax4.plot(daily_sales.index, daily_sales.values, linewidth=1, alpha=0.8, color='darkblue')
            ax4.set_xlabel('Fecha')
            ax4.set_ylabel('Ventas (Media móvil 7d)')
            ax4.set_title('Evolución de Ventas\n(Training Set)', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Top ciudades con tiendas - DERECHA
        ax5 = fig.add_subplot(gs[2, 1])
        if 'stores' in self.datasets:
            stores_df = self.datasets['stores'].copy()
            top_cities = stores_df['city'].value_counts().head(8)
            ax5.barh(range(len(top_cities)), top_cities.values, 
                    color=sns.color_palette(self.color_palette, len(top_cities)))
            ax5.set_xlabel('Número de Tiendas')
            ax5.set_ylabel('Ciudad')
            ax5.set_title('Top 8 Ciudades\n(Número de Tiendas)', fontweight='bold')
            ax5.set_yticks(range(len(top_cities)))
            ax5.set_yticklabels(top_cities.index, fontsize=8)
        
        # 6. Correlaciones principales - ABAJO COMPLETO
        ax6 = fig.add_subplot(gs[3, :])
        if 'train' in self.datasets:
            numeric_cols = self.datasets['train'].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Tomar muestra para eficiencia
                sample_size = min(50000, len(self.datasets['train']))
                train_sample = self.datasets['train'].sample(sample_size, random_state=42)
                corr_matrix = train_sample[numeric_cols].corr()
                
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                           center=0, square=True, ax=ax6, cbar_kws={"shrink": .8},
                           annot_kws={'size': 9})
                ax6.set_title('Matriz de Correlación - Variables Numéricas (Training Set)', 
                             fontweight='bold', pad=15)
        
        plt.suptitle('Análisis Exploratorio General - Corporación Favorita', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def analyze_sales_patterns(self, figsize: Tuple[int, int] = (8.5, 11)) -> plt.Figure:
        """
        Analiza patrones de ventas específicos del dataset de Favorita.
        Formato carta optimizado.
        """
        if 'train' not in self.datasets:
            print("Dataset de entrenamiento no disponible.")
            return None
        
        train_df = self.datasets['train'].copy()
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df['year'] = train_df['date'].dt.year
        train_df['month'] = train_df['date'].dt.month
        train_df['weekday'] = train_df['date'].dt.dayofweek
        train_df['quarter'] = train_df['date'].dt.quarter
        
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        axes = axes.flatten()
        
        # 1. Tendencia temporal de ventas
        monthly_sales = train_df.groupby('date')['unit_sales'].sum().rolling(30).mean()
        axes[0].plot(monthly_sales.index, monthly_sales.values, linewidth=1.2, color='navy')
        axes[0].set_title('A) Tendencia de Ventas\n(Media Móvil 30 días)', fontweight='bold', fontsize=10)
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Ventas Promedio')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Estacionalidad por mes
        monthly_pattern = train_df.groupby('month')['unit_sales'].mean()
        axes[1].bar(monthly_pattern.index, monthly_pattern.values, 
                   color=sns.color_palette(self.color_palette, 12))
        axes[1].set_title('B) Patrón Estacional\nMensual', fontweight='bold', fontsize=10)
        axes[1].set_xlabel('Mes')
        axes[1].set_ylabel('Ventas Promedio')
        axes[1].set_xticks(range(1, 13))
        
        # 3. Patrón semanal
        weekday_names = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
        weekly_pattern = train_df.groupby('weekday')['unit_sales'].mean()
        axes[2].bar(weekly_pattern.index, weekly_pattern.values, 
                   color=sns.color_palette(self.color_palette, 7))
        axes[2].set_title('C) Patrón Semanal', fontweight='bold', fontsize=10)
        axes[2].set_xlabel('Día de la Semana')
        axes[2].set_ylabel('Ventas Promedio')
        axes[2].set_xticks(range(7))
        axes[2].set_xticklabels(weekday_names)
        
        # 4. Top 8 productos por ventas
        product_sales = train_df.groupby('item_nbr')['unit_sales'].sum().sort_values(ascending=False)
        top_products = product_sales.head(8)
        axes[3].barh(range(len(top_products)), top_products.values,
                    color=sns.color_palette("viridis", len(top_products)))
        axes[3].set_title('D) Top 8 Productos\npor Ventas Totales', fontweight='bold', fontsize=10)
        axes[3].set_xlabel('Ventas Totales')
        axes[3].set_ylabel('Ranking')
        axes[3].set_yticks(range(len(top_products)))
        axes[3].set_yticklabels([f'P{i}' for i in top_products.index], fontsize=8)
        
        # 5. Top 8 tiendas por ventas
        store_sales = train_df.groupby('store_nbr')['unit_sales'].sum().sort_values(ascending=False)
        top_stores = store_sales.head(8)
        axes[4].bar(range(len(top_stores)), top_stores.values, 
                   color=sns.color_palette("plasma", 8))
        axes[4].set_title('E) Top 8 Tiendas\npor Ventas Totales', fontweight='bold', fontsize=10)
        axes[4].set_xlabel('Ranking de Tienda')
        axes[4].set_ylabel('Ventas Totales')
        axes[4].set_xticks(range(len(top_stores)))
        axes[4].set_xticklabels([f'T{i}' for i in top_stores.index], rotation=45)
        
        # 6. Distribución de ventas unitarias (log scale)
        sales_sample = train_df.loc[
    train_df['unit_sales'] <= train_df['unit_sales'].quantile(0.98),
    'unit_sales'
].sample(min(10000, len(train_df)), random_state=42)
        sales_positive = sales_sample[sales_sample > 0]  # Solo valores positivos para log
        axes[5].hist(sales_positive, bins=17, alpha=0.7, color='skyblue', edgecolor='black')
        axes[5].set_title('F) Distribución de\nVentas Unitarias', fontweight='bold', fontsize=10)
        axes[5].set_xlabel('Ventas Unitarias')
        axes[5].set_ylabel('Frecuencia')
        #axes[5].set_xscale('log')
        
        # 7. Ventas por año
        yearly_sales = train_df.groupby('year')['unit_sales'].sum()
        axes[6].bar(yearly_sales.index, yearly_sales.values, 
                   color=sns.color_palette(self.color_palette, len(yearly_sales)))
        axes[6].set_title('G) Ventas Totales\npor Año', fontweight='bold', fontsize=10)
        axes[6].set_xlabel('Año')
        axes[6].set_ylabel('Ventas Totales')
        
        # 8. Heatmap simplificado - ventas por mes y día de la semana
        heatmap_data = train_df.groupby(['month', 'weekday'])['unit_sales'].mean().unstack()
        im = axes[7].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        axes[7].set_title('H) Heatmap: Ventas\npor Mes y Día', fontweight='bold', fontsize=10)
        axes[7].set_xlabel('Día de la Semana')
        axes[7].set_ylabel('Mes')
        axes[7].set_xticks(range(7))
        axes[7].set_xticklabels(weekday_names, fontsize=8)
        axes[7].set_yticks(range(0, 12, 2))
        axes[7].set_yticklabels(range(1, 13, 2))
        cbar = plt.colorbar(im, ax=axes[7], shrink=0.6)
        cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle('Análisis Detallado de Patrones de Ventas - Corporación Favorita', 
                     fontsize=13, fontweight='bold', y=0.98)
        
        return fig
    
    def analyze_external_factors(self, figsize: Tuple[int, int] = (8.5, 11)) -> plt.Figure:
        """
        Analiza factores externos como días festivos, precios del petróleo, etc.
        Formato carta optimizado.
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        axes = axes.flatten()
        
        # 1. Análisis de precios del petróleo
        if 'oil' in self.datasets:
            oil_df = self.datasets['oil'].copy()
            oil_df['date'] = pd.to_datetime(oil_df['date'])
            oil_df = oil_df.sort_values('date').dropna()
            
            axes[0].plot(oil_df['date'], oil_df['dcoilwtico'], linewidth=1.2, color='darkred')
            axes[0].set_title('A) Evolución del\nPrecio del Petróleo', fontweight='bold', fontsize=10)
            axes[0].set_xlabel('Fecha')
            axes[0].set_ylabel('Precio WTI (USD)')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Correlación con ventas si está disponible
            if 'train' in self.datasets:
                train_df = self.datasets['train'].copy()
                train_df['date'] = pd.to_datetime(train_df['date'])
                daily_sales = train_df.groupby('date')['unit_sales'].sum()
                
                # Merge con precios del petróleo
                merged_data = pd.merge(daily_sales.reset_index(), oil_df, on='date', how='inner')
                if len(merged_data) > 0 and not merged_data['dcoilwtico'].isna().all():
                    correlation = merged_data['unit_sales'].corr(merged_data['dcoilwtico'])
                    axes[0].text(0.02, 0.98, f'Corr: {correlation:.3f}', 
                               transform=axes[0].transAxes, va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Análisis de días festivos
        if 'holidays_events' in self.datasets:
            holidays_df = self.datasets['holidays_events'].copy()
            holiday_types = holidays_df['type'].value_counts()
            
            axes[1].pie(holiday_types.values, labels=holiday_types.index, autopct='%1.1f%%',
                       colors=sns.color_palette(self.color_palette, len(holiday_types)),
                       textprops={'fontsize': 8})
            axes[1].set_title('B) Tipos de\nDías Festivos', fontweight='bold', fontsize=10)
        
        # 3. Transacciones por tienda
        if 'transactions' in self.datasets:
            trans_df = self.datasets['transactions'].copy()
            store_transactions = trans_df.groupby('store_nbr')['transactions'].mean()
            
            axes[2].hist(store_transactions.values, bins=30, alpha=0.7, 
                        color='lightgreen', edgecolor='black')
            axes[2].set_title('C) Distribución de\nTransacciones por Tienda', fontweight='bold', fontsize=10)
            axes[2].set_xlabel('Transacciones Promedio')
            axes[2].set_ylabel('Número de Tiendas')
        
        # 4. Top ciudades por número de tiendas
        if 'stores' in self.datasets:
            stores_df = self.datasets['stores'].copy()
            city_counts = stores_df['city'].value_counts().head(8)
            axes[3].barh(range(len(city_counts)), city_counts.values,
                        color=sns.color_palette("Set3", len(city_counts)))
            axes[3].set_title('D) Top 8 Ciudades\npor Número de Tiendas', fontweight='bold', fontsize=10)
            axes[3].set_xlabel('Número de Tiendas')
            axes[3].set_yticks(range(len(city_counts)))
            axes[3].set_yticklabels(city_counts.index, fontsize=8)
        
        # 5. Tipos de tiendas
        if 'stores' in self.datasets:
            store_types = stores_df['type'].value_counts()
            axes[4].bar(store_types.index, store_types.values,
                       color=sns.color_palette(self.color_palette, len(store_types)))
            axes[4].set_title('E) Distribución por\nTipo de Tienda', fontweight='bold', fontsize=10)
            axes[4].set_xlabel('Tipo de Tienda')
            axes[4].set_ylabel('Número de Tiendas')
            axes[4].tick_params(axis='x', rotation=45)
        
        # 6. Top familias de productos
        if 'items' in self.datasets:
            items_df = self.datasets['items'].copy()
            family_counts = items_df['family'].value_counts().head(10)
            
            axes[5].barh(range(len(family_counts)), family_counts.values,
                        color=sns.color_palette("tab10", len(family_counts)))
            axes[5].set_title('F) Top 10 Familias\nde Productos', fontweight='bold', fontsize=10)
            axes[5].set_xlabel('Número de Productos')
            axes[5].set_yticks(range(len(family_counts)))
            axes[5].set_yticklabels(family_counts.index, fontsize=7)
        
        plt.suptitle('Análisis de Factores Externos y Características del Negocio', 
                     fontsize=13, fontweight='bold', y=0.98)
        
        return fig
    
    def statistical_summary_report(self) -> pd.DataFrame:
        """
        Genera un reporte estadístico detallado de los datasets.
        
        Returns:
        --------
        pd.DataFrame
            Resumen estadístico completo
        """
        if not self.datasets:
            print("No hay datasets cargados.")
            return None
        
        summary_data = []
        
        for dataset_name, df in self.datasets.items():
            # Estadísticas básicas
            n_rows, n_cols = df.shape
            n_numeric = len(df.select_dtypes(include=[np.number]).columns)
            n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
            n_datetime = len(df.select_dtypes(include=['datetime64']).columns)
            
            # Valores nulos
            total_nulls = df.isnull().sum().sum()
            null_percentage = (total_nulls / (n_rows * n_cols)) * 100
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            
            summary_data.append({
                'Dataset': dataset_name,
                'Filas': n_rows,
                'Columnas': n_cols,
                'Numéricas': n_numeric,
                'Categóricas': n_categorical,
                'Fecha/Hora': n_datetime,
                'Valores_Nulos': total_nulls,
                'Porcentaje_Nulos': f"{null_percentage:.2f}%",
                'Memoria_MB': f"{memory_usage:.2f}",
                'Periodo': self._get_date_range(df) if dataset_name in ['train', 'test', 'oil', 'transactions'] else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*80)
        print("REPORTE ESTADÍSTICO DETALLADO - CORPORACIÓN FAVORITA")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Obtiene el rango de fechas de un dataset."""
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            try:
                dates = pd.to_datetime(df[date_cols[0]].dropna())
                return f"{dates.min().strftime('%Y-%m-%d')} a {dates.max().strftime('%Y-%m-%d')}"
            except:
                return "N/A"
        return "N/A"

# Resto del código de implementación...
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar rutas
BASE_DIR = Path.cwd()  # Directorio actual
DATA_DIR = BASE_DIR / ".data" / "raw_data"
FIGURES_DIR = BASE_DIR / "figures" / "favorita_analysis"

# Crear directorio de figuras si no existe
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("📁 Configuración de rutas:")
print(f"   • Datos: {DATA_DIR}")
print(f"   • Figuras: {FIGURES_DIR}")
print("=" * 60)

def verify_data_files(data_path):
    """
    Verifica que todos los archivos necesarios estén presentes.
    """
    required_files = {
        'train': 'train.csv',
        'test': 'test.csv', 
        'stores': 'stores.csv',
        'items': 'items.csv',
        'transactions': 'transactions.csv',
        'oil': 'oil.csv',
        'holidays_events': 'holidays_events.csv'
    }
    
    file_status = {}
    data_path = Path(data_path)
    
    print("🔍 Verificando archivos...")
    for name, filename in required_files.items():
        filepath = data_path / filename
        exists = filepath.exists()
        file_status[name] = {
            'filename': filename,
            'exists': exists,
            'path': filepath,
            'size_mb': filepath.stat().st_size / (1024*1024) if exists else 0
        }
        
        status_icon = "✅" if exists else "❌"
        size_info = f"({file_status[name]['size_mb']:.1f} MB)" if exists else "(no encontrado)"
        print(f"   {status_icon} {filename} {size_info}")
    
    missing_files = [name for name, status in file_status.items() if not status['exists']]
    if missing_files:
        print(f"\n⚠️  Archivos faltantes: {', '.join(missing_files)}")
    else:
        print("\n🎉 Todos los archivos están presentes!")
    
    return file_status

def save_figure_safely(fig, filepath_base):
    """
    Guarda una figura de forma segura en PNG y PDF.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figura a guardar
    filepath_base : str or Path
        Ruta base sin extensión
    """
    try:
        # Guardar PNG
        png_path = f"{filepath_base}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"   ✓ Guardado: {os.path.basename(pdf_path)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error guardando {filepath_base}: {str(e)}")
        return False

def main_favorita_analysis():
    """
    Función principal para ejecutar el análisis completo de Corporación Favorita.
    """
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO - CORPORACIÓN FAVORITA")
    print("=" * 60)
    
    # 1. Verificar archivos
    file_status = verify_data_files(DATA_DIR)
    
    # 2. Contar archivos disponibles
    available_files = sum(1 for status in file_status.values() if status['exists'])
    total_files = len(file_status)
    
    if available_files == 0:
        print("❌ No se encontraron archivos de datos. Verifique la ruta:", DATA_DIR)
        return None
    
    print(f"\n📊 Archivos disponibles: {available_files}/{total_files}")
    
    # 3. Inicializar análisis EDA
    try:
        eda = FavoritaEDA(str(DATA_DIR), color_palette="Set2")
        
        # 4. Cargar datasets
        print("\n" + "="*60)
        datasets = eda.load_favorita_datasets()
        
        if not datasets:
            print("❌ No se pudieron cargar los datasets.")
            return None
            
        # 5. Generar reporte estadístico
        print("\n" + "="*60)
        print("📈 GENERANDO REPORTE ESTADÍSTICO...")
        summary_report = eda.statistical_summary_report()
        
        # 6. Ejecutar análisis visual completo
        print("\n" + "="*60)
        print("🎨 GENERANDO VISUALIZACIONES...")
        
        try:
            # Resumen general
            print("   • Generando resumen general...")
            fig1 = eda.comprehensive_data_overview(figsize=(8.5, 11))
            if fig1:
                save_figure_safely(fig1, FIGURES_DIR / "01_resumen_general")
                plt.show()
                plt.close(fig1)  # Liberar memoria
            
            # Patrones de ventas
            if 'train' in datasets:
                print("   • Analizando patrones de ventas...")
                fig2 = eda.analyze_sales_patterns(figsize=(8.5, 11))
                if fig2:
                    save_figure_safely(fig2, FIGURES_DIR / "02_patrones_ventas")
                    plt.show()
                    plt.close(fig2)  # Liberar memoria
            
            # Factores externos
            print("   • Analizando factores externos...")
            fig3 = eda.analyze_external_factors(figsize=(8.5, 11))
            if fig3:
                save_figure_safely(fig3, FIGURES_DIR / "03_factores_externos")
                plt.show()
                plt.close(fig3)  # Liberar memoria
            
        except Exception as e:
            print(f"⚠️  Error en visualizaciones: {str(e)}")
            print("   Continuando con análisis básico...")
        
        # 7. Generar análisis adicional por dataset individual
        print("\n" + "="*60)
        print("🔬 ANÁLISIS DETALLADO POR DATASET...")
        
        for dataset_name, df in datasets.items():
            try:
                print(f"\n📋 Analizando: {dataset_name.upper()}")
                analyze_individual_dataset(df, dataset_name, FIGURES_DIR)
            except Exception as e:
                print(f"   ⚠️  Error en {dataset_name}: {str(e)}")
        
        # 8. Generar resumen final
        print("\n" + "="*60)
        print("📋 RESUMEN FINAL DEL ANÁLISIS")
        print("="*60)
        
        total_rows = sum(df.shape[0] for df in datasets.values())
        total_cols = sum(df.shape[1] for df in datasets.values())
        
        print(f"✅ Datasets analizados: {len(datasets)}")
        print(f"📊 Total de registros: {total_rows:,}")
        print(f"🔢 Total de columnas: {total_cols}")
        print(f"💾 Archivos guardados en: {FIGURES_DIR}")
        
        # Lista de archivos generados
        generated_files = list(FIGURES_DIR.glob("*.png")) + list(FIGURES_DIR.glob("*.pdf"))
        if generated_files:
            print(f"🖼️  Figuras generadas: {len(generated_files)}")
            for file in sorted(generated_files)[:6]:  # Mostrar solo las primeras 6
                print(f"   • {file.name}")
            if len(generated_files) > 6:
                print(f"   ... y {len(generated_files) - 6} más")
        
        print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
        
        return eda
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        import traceback
        print("\nDetalles del error:")
        traceback.print_exc()
        return None

def analyze_individual_dataset(df, dataset_name, save_dir):
    """
    Realiza análisis específico para cada dataset individual.
    """
    # Información básica
    print(f"   📏 Dimensiones: {df.shape[0]:,} × {df.shape[1]}")
    
    # Análisis por tipo de dataset
    if dataset_name == 'train':
        analyze_train_dataset(df, save_dir)
    elif dataset_name == 'stores':
        analyze_stores_dataset(df, save_dir)
    elif dataset_name == 'items':
        analyze_items_dataset(df, save_dir)
    elif dataset_name == 'transactions':
        analyze_transactions_dataset(df, save_dir)
    elif dataset_name == 'oil':
        analyze_oil_dataset(df, save_dir)
    elif dataset_name == 'holidays_events':
        analyze_holidays_dataset(df, save_dir)
    
    # Análisis general de valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠️  Valores nulos encontrados: {null_counts.sum():,}")

def analyze_train_dataset(df, save_dir):
    """Análisis específico del dataset de entrenamiento."""
    print("   🎯 Analizando dataset de entrenamiento...")
    
    # Estadísticas de ventas
    if 'unit_sales' in df.columns:
        sales_stats = df['unit_sales'].describe()
        print(f"      • Ventas promedio: {sales_stats['mean']:.2f}")
        print(f"      • Ventas máximas: {sales_stats['max']:.2f}")
        
        # Distribución de ventas negativas
        negative_sales = (df['unit_sales'] < 0).sum()
        if negative_sales > 0:
            print(f"      ⚠️  Ventas negativas: {negative_sales:,} ({negative_sales/len(df)*100:.1f}%)")

def analyze_stores_dataset(df, save_dir):
    """Análisis específico del dataset de tiendas."""
    print("   🏪 Analizando dataset de tiendas...")
    
    if 'city' in df.columns:
        n_cities = df['city'].nunique()
        print(f"      • Ciudades únicas: {n_cities}")
    
    if 'type' in df.columns:
        store_types = df['type'].value_counts()
        print(f"      • Tipos de tienda: {len(store_types)}")

def analyze_items_dataset(df, save_dir):
    """Análisis específico del dataset de productos."""
    print("   📦 Analizando dataset de productos...")
    
    if 'family' in df.columns:
        n_families = df['family'].nunique()
        print(f"      • Familias de productos: {n_families}")
    
    if 'class' in df.columns:
        n_classes = df['class'].nunique()
        print(f"      • Clases de productos: {n_classes}")

def analyze_transactions_dataset(df, save_dir):
    """Análisis específico del dataset de transacciones."""
    print("   💳 Analizando dataset de transacciones...")
    
    if 'transactions' in df.columns:
        trans_stats = df['transactions'].describe()
        print(f"      • Transacciones promedio: {trans_stats['mean']:.0f}")

def analyze_oil_dataset(df, save_dir):
    """Análisis específico del dataset de petróleo."""
    print("   🛢️  Analizando dataset de petróleo...")
    
    if 'dcoilwtico' in df.columns:
        oil_stats = df['dcoilwtico'].describe()
        null_values = df['dcoilwtico'].isnull().sum()
        print(f"      • Precio promedio: ${oil_stats['mean']:.2f}")
        if null_values > 0:
            print(f"      ⚠️  Valores faltantes: {null_values}")

def analyze_holidays_dataset(df, save_dir):
    """Análisis específico del dataset de días festivos."""
    print("   🎉 Analizando dataset de días festivos...")
    
    if 'type' in df.columns:
        holiday_types = df['type'].value_counts()
        print(f"      • Tipos de días festivos: {len(holiday_types)}")

if __name__ == "__main__":
    # Ejecutar análisis principal
    eda_result = main_favorita_analysis()
    
    # Mensaje final
    if eda_result:
        print("\n" + "="*60)
        print("🎯 PRÓXIMOS PASOS SUGERIDOS:")
        print("="*60)
        print("1. 📊 Revisar las figuras generadas en:", FIGURES_DIR)
        print("2. 🔍 Analizar correlaciones encontradas")
        print("3. 🧹 Limpiar datos basado en insights obtenidos")
        print("4. 🤖 Desarrollar modelos de forecasting")
        print("5. 📝 Documentar findings para paper")
    else:
        print("\n❌ El análisis no se completó correctamente.")
        print("   Revise los errores mostrados arriba.")
        
        # Guardar PDF
        pdf_path = f"{filepath_base}.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"   ✓ Guardado: {os.path.basename(pdf_path)}")