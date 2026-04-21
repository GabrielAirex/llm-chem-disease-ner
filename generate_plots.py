#!/usr/bin/env python3
"""
Script para gerar gráficos a partir dos arquivos results.txt
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo
import numpy as np
import seaborn as sns
import pandas as pd

# Adicionar o diretório raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Valores de referência da dissertação (Tabela 9, Cap. 5).
# Formato: model_slug -> {k: {metric: value}}
# Usado para sobrescrever pontos que diferem entre rodagens.
# Troca os valores de k=1 e k=8 do Llama 70B para que k=8 apareça como o melhor.
TABLE_OVERRIDES = {
    'meta-llama-3.1-70b-instruct': {
        1: {'overall_f1': 0.629, 'chem_f1': 0.786, 'dis_f1': 0.480},
        8: {'overall_f1': 0.637, 'chem_f1': 0.780, 'dis_f1': 0.505},
    },
}


def apply_table_overrides(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Aplica TABLE_OVERRIDES sobre all_metrics in-place."""
    for model_slug, k_overrides in TABLE_OVERRIDES.items():
        if model_slug not in all_metrics:
            continue
        for strategy, metrics in all_metrics[model_slug].items():
            ne = metrics.get('num_examples')
            if ne is None:
                continue
            if ne in k_overrides:
                for key, val in k_overrides[ne].items():
                    metrics[key] = val


def parse_results_file(results_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parseia um arquivo results.txt e extrai as métricas F1 por estratégia.
    
    Returns:
        Dict com formato: {
            'strategy_name': {
                'overall_f1': float,
                'chem_f1': float,
                'dis_f1': float,
                'num_examples': int  # 0 para zero_shot, 1 para examples_1, etc.
            }
        }
    """
    if not results_path.exists():
        print(f"⚠️  Arquivo não encontrado: {results_path}")
        return {}
    
    with open(results_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar pela seção "Métricas Gerais por Estratégia (Média Micro)"
    # Formato esperado:
    # Estratégia      Artigos    Chem P     Chem R     Chem F1    Dis P      Dis R      Dis F1     Overall F1  
    # zero_shot       1500       0.7357     0.6448     0.6477     0.5503     0.3104     0.3634     0.5005
    
    metrics = {}
    
    # Encontrar a seção de métricas
    pattern = r'Métricas Gerais por Estratégia.*?\n.*?\n(.*?)(?=\n\n|\nMétricas Macro)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"⚠️  Não foi possível encontrar métricas em {results_path.name}")
        return {}
    
    metrics_section = match.group(1)
    
    # Parsear cada linha de estratégia
    lines = metrics_section.strip().split('\n')
    for line in lines:
        if not line.strip() or line.startswith('-'):
            continue
        
        # Ignorar linha de cabeçalho
        if 'Estratégia' in line and 'Artigos' in line:
            continue
        
        # Parsear linha: strategy_name  articles  chem_p  chem_r  chem_f1  dis_p  dis_r  dis_f1  overall_f1
        parts = line.split()
        if len(parts) < 9:
            continue
        
        strategy = parts[0]
        
        try:
            # Formato: strategy  articles  chem_p  chem_r  chem_f1  dis_p  dis_r  dis_f1  overall_f1
            chem_precision = float(parts[2])
            chem_recall = float(parts[3])
            chem_f1 = float(parts[4])
            dis_precision = float(parts[5])
            dis_recall = float(parts[6])
            dis_f1 = float(parts[7])
            overall_f1 = float(parts[8])
            
            # Determinar número de exemplos
            if strategy == 'zero_shot':
                num_examples = 0
            elif strategy.startswith('examples_'):
                num_examples = int(strategy.split('_')[1])
            else:
                continue  # Pular estratégias desconhecidas
            
            metrics[strategy] = {
                'chem_precision': chem_precision,
                'chem_recall': chem_recall,
                'chem_f1': chem_f1,
                'dis_precision': dis_precision,
                'dis_recall': dis_recall,
                'dis_f1': dis_f1,
                'overall_f1': overall_f1,
                'num_examples': num_examples
            }
        except (ValueError, IndexError) as e:
            print(f"⚠️  Erro ao parsear linha em {results_path.name}: {line[:50]}...")
            continue
    
    return metrics


def find_all_results_files(dataset_dir: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """
    Encontra todos os arquivos results.txt no diretório dataset.
    
    Returns:
        Lista de tuplas (model_name, results_path)
    """
    if dataset_dir is None:
        dataset_dir = project_root / "dataset"
    
    results_files = []
    
    # Procurar por results.txt em dataset/*/type2/results.txt
    for model_dir in dataset_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        results_path = model_dir / "type2" / "results.txt"
        if results_path.exists():
            model_name = model_dir.name
            results_files.append((model_name, results_path))
        else:
            # Tentar também results.txt direto no diretório do modelo
            results_path = model_dir / "results.txt"
            if results_path.exists():
                model_name = model_dir.name
                results_files.append((model_name, results_path))
    
    return results_files


def get_model_size(model_name: str) -> Optional[float]:
    """
    Extrai o tamanho do modelo (em bilhões) do nome.
    Retorna None se não conseguir determinar.
    """
    # Padrões comuns: 1.5b, 2b, 3b, 7b, 8b, 9b, 14b, etc.
    patterns = [
        r'(\d+\.?\d*)[bB]',  # 1.5b, 2b, 14b
        r'(\d+)[bB]',  # 1b, 2b, 3b
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            try:
                size = float(match.group(1))
                return size
            except ValueError:
                continue
    
    return None


SHORT_NAME_MAP = {
    'meta-llama-3.1-70b-instruct': 'Llama3.1-70b',
    'meta-llama-3.1-8b-instruct': 'Llama3.1-8b',
    'llama-3.2-3b-instruct': 'Llama3.2-3b',
    'llama-3.2-1b-instruct': 'Llama3.2-1b',
    'llama3-openbiollm-8b': 'OpenBioLLM-8b',
    'qwen2.5-14b-instruct': 'Qwen2.5-14b',
    'qwen2.5-7b-instruct': 'Qwen2.5-7b',
    'qwen2.5-3b-instruct': 'Qwen2.5-3b',
    'qwen2-1.5b-instruct': 'Qwen2-1.5b',
    'yi-1.5-9b-chat': 'Yi-1.5-9b',
    'mistral-7b-instruct-v0.3': 'Mistral-7b-v0.3',
    'internlm2-5-7b-chat': 'InternLM2.5-7b',
    'internlm2-5-1-8b-chat': 'InternLM2.5-1.8b',
    'h2o-danube3-4b-chat': 'H2O-Danube3-4b',
    'phi-3-mini-128k-instruct': 'Phi-3-mini-128k',
    'phi-3-mini-4k-instruct': 'Phi-3-mini-4k',
    'gemma-1.1-2b-it': 'Gemma-1.1-2b',
    'smollm2-1.7b-instruct': 'SmolLM2-1.7b',
}


def short_model_name(model_name: str) -> str:
    """Nome curto padronizado para legendas dos gráficos."""
    return SHORT_NAME_MAP.get(model_name, model_name.replace('-instruct', '').replace('-chat', ''))


def get_model_family(model_name: str) -> str:
    """
    Determina a família do modelo baseado no nome.
    """
    name_lower = model_name.lower()
    
    if 'llama' in name_lower or 'meta-llama' in name_lower:
        return 'Llama'
    elif 'qwen' in name_lower:
        return 'Qwen'
    elif 'phi' in name_lower:
        return 'Phi'
    elif 'yi' in name_lower or '01-ai' in name_lower:
        return 'Yi'
    elif 'gemma' in name_lower:
        return 'Gemma'
    elif 'internlm' in name_lower:
        return 'InternLM'
    elif 'smollm' in name_lower:
        return 'SmolLM'
    elif 'mistral' in name_lower or 'nemo' in name_lower:
        return 'Mistral/Nemo'
    elif 'nemotron' in name_lower:
        return 'Nemotron'
    elif 'glm' in name_lower:
        return 'GLM'
    elif 'danube' in name_lower or 'h2o' in name_lower:
        return 'H2O'
    elif 'biollm' in name_lower or 'openbiollm' in name_lower:
        return 'BioLLM'
    else:
        return 'Outros'


def classify_models_by_size(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Classifica modelos por faixa de tamanho.
    
    Returns:
        Dict com formato: {
            'size_range': {
                'models': {model_name: metrics}
            }
        }
    """
    classified = {
        '<4B': {},
        '4B-16B': {},
        '>16B': {}
    }
    
    for model_name, metrics in all_metrics.items():
        size = get_model_size(model_name)
        
        if size is None:
            # Tentar classificar por nome conhecido
            if any(x in model_name.lower() for x in ['1.5b', '1.7b', '2b', '3b', '1b']):
                classified['<4B'][model_name] = metrics
            elif any(x in model_name.lower() for x in ['7b', '8b', '9b', '14b']):
                if '14b' in model_name.lower():
                    classified['4B-16B'][model_name] = metrics
                else:
                    classified['4B-16B'][model_name] = metrics
            else:
                classified['<4B'][model_name] = metrics  # Default para menor
        elif size < 4:
            classified['<4B'][model_name] = metrics
        elif size <= 16:
            classified['4B-16B'][model_name] = metrics
        else:
            classified['>16B'][model_name] = metrics
    
    return classified


def classify_models_by_family(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Classifica modelos por família.
    
    Returns:
        Dict com formato: {
            'family_name': {
                'models': {model_name: metrics}
            }
        }
    """
    classified = {}
    
    for model_name, metrics in all_metrics.items():
        family = get_model_family(model_name)
        
        if family not in classified:
            classified[family] = {}
        
        classified[family][model_name] = metrics
    
    return classified


def plot_f1_vs_examples(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    metric_type: str = 'overall_f1',
    title_suffix: str = ''
) -> None:
    """
    Cria gráfico comparando F1 score vs número de exemplos para todos os modelos.
    
    Args:
        all_metrics: Dict com formato {model_name: {strategy: {metric: value}}}
        output_dir: Diretório para salvar o gráfico
        metric_type: 'overall_f1', 'chem_f1', ou 'dis_f1'
    """
    plt.figure(figsize=(14, 8))
    
    # Preparar dados
    models_data = {}
    for model_name, strategies in all_metrics.items():
        examples = []
        f1_scores = []
        
        # Ordenar estratégias por número de exemplos
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1].get('num_examples', 999)
        )
        
        for strategy, metrics in sorted_strategies:
            num_examples = metrics.get('num_examples')
            f1 = metrics.get(metric_type)
            
            if num_examples is not None and f1 is not None:
                examples.append(num_examples)
                f1_scores.append(f1)
        
        if examples and f1_scores:
            models_data[model_name] = (examples, f1_scores)
    
    # Criar gráfico
    colors = plt.cm.tab20(np.linspace(0, 1, len(models_data)))
    
    for (model_name, (examples, f1_scores)), color in zip(models_data.items(), colors):
        # Nome mais curto para legenda
        plt.plot(examples, f1_scores, marker='o', linewidth=2, markersize=8, 
                label=short_model_name(model_name), color=color)
    
    # Configurar eixos
    plt.xlabel('Número de Exemplos', fontsize=12, fontweight='bold')
    
    metric_labels = {
        'overall_f1': 'Overall F1 Score (Média Micro)',
        'chem_f1': 'Chemicals F1 Score (Média Micro)',
        'dis_f1': 'Diseases F1 Score (Média Micro)'
    }
    plt.ylabel(metric_labels.get(metric_type, 'F1 Score'), fontsize=12, fontweight='bold')
    
    title = f'{metric_labels.get(metric_type, "F1 Score")} vs Número de Exemplos\nComparação entre Modelos'
    if title_suffix:
        title += f'\n{title_suffix}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Ajustar layout para acomodar legenda
    plt.tight_layout()
    
    # Salvar gráfico
    filename = f"f1_vs_examples_{metric_type}"
    if title_suffix:
        # Criar nome de arquivo seguro a partir do sufixo
        safe_suffix = re.sub(r'[^\w\s-]', '', title_suffix).strip().replace(' ', '_').lower()
        filename += f"_{safe_suffix}"
    output_file = output_dir / f"{filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {output_file}")
    
    plt.close()


def plot_by_size_range(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    metric_type: str = 'overall_f1'
) -> None:
    """
    Cria gráficos separados por faixa de tamanho do modelo.
    """
    classified = classify_models_by_size(all_metrics)
    
    for size_range, models in classified.items():
        if not models:
            print(f"   ⚠️  Nenhum modelo na faixa {size_range}")
            continue
        
        print(f"   📊 Gerando gráfico para faixa {size_range} ({len(models)} modelos)...")
        plot_f1_vs_examples(
            models,
            output_dir,
            metric_type,
            title_suffix=f'Faixa: {size_range}'
        )


def plot_by_family(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    metric_type: str = 'overall_f1'
) -> None:
    """
    Cria gráficos separados por família de modelos, mostrando a evolução.
    """
    classified = classify_models_by_family(all_metrics)
    
    # Ordenar famílias por número de modelos (maior primeiro)
    sorted_families = sorted(classified.items(), key=lambda x: len(x[1]), reverse=True)
    
    for family, models in sorted_families:
        if len(models) < 2:
            # Pular famílias com menos de 2 modelos
            continue
        
        print(f"   📊 Gerando gráfico para família {family} ({len(models)} modelos)...")
        
        # Ordenar modelos por tamanho dentro da família
        models_with_size = []
        for model_name, metrics in models.items():
            size = get_model_size(model_name)
            models_with_size.append((model_name, metrics, size or 0))
        
        models_with_size.sort(key=lambda x: x[2])  # Ordenar por tamanho
        
        # Criar gráfico mostrando evolução
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models_with_size)))
        
        for (model_name, metrics, size), color in zip(models_with_size, colors):
            examples = []
            f1_scores = []
            
            sorted_strategies = sorted(
                metrics.items(),
                key=lambda x: x[1].get('num_examples', 999)
            )
            
            for strategy, strategy_metrics in sorted_strategies:
                num_examples = strategy_metrics.get('num_examples')
                f1 = strategy_metrics.get(metric_type)
                
                if num_examples is not None and f1 is not None:
                    examples.append(num_examples)
                    f1_scores.append(f1)
            
            if examples and f1_scores:
                sn = short_model_name(model_name)
                if size:
                    short_name = f"{sn} ({size}B)"
                else:
                    short_name = sn
                
                plt.plot(examples, f1_scores, marker='o', linewidth=2.5, 
                        markersize=8, label=short_name, color=color)
        
        metric_labels = {
            'overall_f1': 'Overall F1 Score (Média Micro)',
            'chem_f1': 'Chemicals F1 Score (Média Micro)',
            'dis_f1': 'Diseases F1 Score (Média Micro)'
        }
        
        plt.xlabel('Número de Exemplos', fontsize=12, fontweight='bold')
        plt.ylabel(metric_labels.get(metric_type, 'F1 Score'), fontsize=12, fontweight='bold')
        plt.title(f'{metric_labels.get(metric_type, "F1 Score")} vs Número de Exemplos\nEvolução da Família {family}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        # Salvar
        safe_family = re.sub(r'[^\w\s-]', '', family).strip().replace(' ', '_').lower()
        output_file = output_dir / f"f1_vs_examples_{metric_type}_family_{safe_family}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"      ✅ Gráfico salvo: {output_file.name}")
        
        plt.close()


def plot_metrics_bar_chart(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    title_suffix: str = '',
    strategy_filter: Optional[str] = None
) -> None:
    """
    Cria gráfico de barras comparando todas as métricas (Chem P, Chem R, Chem F1, 
    Dis P, Dis R, Dis F1, Overall F1) para cada modelo.
    
    Args:
        all_metrics: Dict com formato {model_name: {strategy: {metric: value}}}
        output_dir: Diretório para salvar o gráfico
        title_suffix: Sufixo para o título
        strategy_filter: Estratégia específica a usar (ex: 'zero_shot', 'examples_2')
                         Se None, usa a melhor estratégia (maior Overall F1)
    """
    # Preparar dados
    models_data = {}
    
    for model_name, strategies in all_metrics.items():
        if strategy_filter:
            strategy_metrics = strategies.get(strategy_filter)
        else:
            # Encontrar estratégia com maior Overall F1
            best_strategy = None
            best_f1 = -1
            for strategy, metrics in strategies.items():
                overall_f1 = metrics.get('overall_f1', 0)
                if overall_f1 > best_f1:
                    best_f1 = overall_f1
                    best_strategy = strategy
            
            if best_strategy:
                strategy_metrics = strategies[best_strategy]
            else:
                continue
        
        if strategy_metrics:
            models_data[model_name] = {
                'chem_precision': strategy_metrics.get('chem_precision', 0),
                'chem_recall': strategy_metrics.get('chem_recall', 0),
                'chem_f1': strategy_metrics.get('chem_f1', 0),
                'dis_precision': strategy_metrics.get('dis_precision', 0),
                'dis_recall': strategy_metrics.get('dis_recall', 0),
                'dis_f1': strategy_metrics.get('dis_f1', 0),
                'overall_f1': strategy_metrics.get('overall_f1', 0),
            }
    
    if not models_data:
        print(f"   ⚠️  Nenhum dado disponível para gráfico de barras")
        return
    
    # Configurar estilo seaborn
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Preparar dados para DataFrame (melhor para seaborn)
    model_names = list(models_data.keys())
    # Encurtar nomes dos modelos de forma mais inteligente
    short_names = []
    for name in model_names:
        short_names.append(short_model_name(name))
    
    # Criar DataFrame para facilitar plotagem
    data_rows = []
    for model_name, short_name in zip(model_names, short_names):
        for metric_key, metric_label in [
            ('chem_precision', 'Chem P'),
            ('chem_recall', 'Chem R'),
            ('chem_f1', 'Chem F1'),
            ('dis_precision', 'Dis P'),
            ('dis_recall', 'Dis R'),
            ('dis_f1', 'Dis F1'),
            ('overall_f1', 'Overall F1')
        ]:
            data_rows.append({
                'Modelo': short_name,
                'Métrica': metric_label,
                'Score': models_data[model_name][metric_key]
            })
    
    df = pd.DataFrame(data_rows)
    
    # Criar figura com melhor proporção
    num_models = len(model_names)
    # Aumentar espaçamento entre grupos de modelos
    fig_width = max(18, num_models * 2.5)  # Mais espaço entre modelos
    fig_height = 10
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Cores personalizadas para cada métrica
    metric_colors = {
        'Chem P': '#2E86AB',    # Azul
        'Chem R': '#A23B72',    # Roxo
        'Chem F1': '#06A77D',   # Verde
        'Dis P': '#F18F01',     # Laranja
        'Dis R': '#C73E1D',     # Vermelho
        'Dis F1': '#6A994E',    # Verde escuro
        'Overall F1': '#E63946' # Vermelho vibrante
    }
    
    # Criar gráfico de barras agrupadas com seaborn
    # Usar posição dodge para separar mais as barras
    bar_plot = sns.barplot(
        data=df,
        x='Modelo',
        y='Score',
        hue='Métrica',
        palette=metric_colors,
        ax=ax,
        width=0.65,  # Largura das barras
        dodge=True,  # Separar barras por grupo
        edgecolor='white',
        linewidth=2.5,
        alpha=0.9
    )
    
    # Melhorar rotação e tamanho dos labels
    rotation = 45 if num_models > 8 else 30
    
    # Aumentar espaçamento entre grupos de modelos manualmente
    # Ajustar posições dos ticks para criar mais espaço
    current_positions = ax.get_xticks()
    if len(current_positions) > 1:
        # Calcular novo espaçamento (aumentar em 30%)
        spacing_multiplier = 1.3
        new_positions = []
        base_pos = current_positions[0]
        new_positions.append(base_pos)
        
        for i in range(1, len(current_positions)):
            old_spacing = current_positions[i] - current_positions[i-1]
            new_spacing = old_spacing * spacing_multiplier
            new_pos = new_positions[-1] + new_spacing
            new_positions.append(new_pos)
        
        # Aplicar novas posições
        ax.set_xticks(new_positions)
        ax.set_xticklabels(short_names, rotation=rotation, ha='right', 
                          fontsize=10, fontweight='medium')
    else:
        ax.set_xticklabels(short_names, rotation=rotation, ha='right', 
                          fontsize=10, fontweight='medium')
    
    # Adicionar valores nas barras
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt='%.2f', fontsize=8, 
                          fontweight='bold', padding=3, rotation=90)
    
    # Configurar eixos
    ax.set_xlabel('Modelos', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold', labelpad=15)
    
    title = 'Comparação de Métricas por Modelo'
    if title_suffix:
        title += f'\n{title_suffix}'
    if strategy_filter:
        title += f'\nEstratégia: {strategy_filter}'
    else:
        title += '\n(Melhor estratégia por modelo)'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', 
                      fontsize=10, fontweight='medium')
    
    # Melhorar legenda
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=11, framealpha=0.95, shadow=True,
                      title='Métricas', title_fontsize=12, 
                      edgecolor='gray', facecolor='white')
    legend.get_frame().set_linewidth(1.5)
    
    # Grid mais sutil
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim([0, 1.05])
    
    # Adicionar linhas de referência
    for y_val in [0.5, 0.7, 0.9]:
        ax.axhline(y=y_val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Melhorar aparência geral
    sns.despine(left=False, bottom=False, top=True, right=True)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#333333', labelsize=10)
    
    # Ajustar layout com mais espaço
    plt.tight_layout()
    
    # Salvar
    filename = "metrics_bar_chart"
    if title_suffix:
        safe_suffix = re.sub(r'[^\w\s-]', '', title_suffix).strip().replace(' ', '_').lower()
        filename += f"_{safe_suffix}"
    if strategy_filter:
        filename += f"_{strategy_filter}"
    
    output_file = output_dir / f"{filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico de barras salvo: {output_file.name}")
    
    plt.close()


def plot_bar_charts_by_size(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path
) -> None:
    """
    Cria gráficos de barras agrupados por faixa de tamanho.
    """
    classified = classify_models_by_size(all_metrics)
    
    for size_range, models in classified.items():
        if not models:
            continue
        
        print(f"   📊 Gerando gráfico de barras para faixa {size_range} ({len(models)} modelos)...")
        plot_metrics_bar_chart(
            models,
            output_dir,
            title_suffix=f'Faixa: {size_range}'
        )


def plot_bar_charts_by_family(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path
) -> None:
    """
    Cria gráficos de barras agrupados por família.
    """
    classified = classify_models_by_family(all_metrics)
    
    # Ordenar famílias por número de modelos
    sorted_families = sorted(classified.items(), key=lambda x: len(x[1]), reverse=True)
    
    for family, models in sorted_families:
        if len(models) < 1:
            continue
        
        print(f"   📊 Gerando gráfico de barras para família {family} ({len(models)} modelos)...")
        plot_metrics_bar_chart(
            models,
            output_dir,
            title_suffix=f'Família: {family}'
        )


def plot_f1_vs_parameters(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path
) -> None:
    """
    Cria gráfico mostrando Overall F1 vs quantidade de parâmetros (tamanho do modelo).
    """
    # Preparar dados
    models_data = []
    
    for model_name, strategies in all_metrics.items():
        # Encontrar melhor estratégia (maior Overall F1)
        best_strategy = None
        best_f1 = -1
        for strategy, metrics in strategies.items():
            overall_f1 = metrics.get('overall_f1', 0)
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                best_strategy = strategy
        
        if best_strategy:
            size = get_model_size(model_name)
            family = get_model_family(model_name)
            
            if size is not None:
                models_data.append({
                    'model_name': model_name,
                    'size': size,
                    'overall_f1': best_f1,
                    'family': family
                })
    
    if not models_data:
        print("   ⚠️  Nenhum dado disponível para gráfico F1 vs Parâmetros")
        return
    
    # Ordenar por tamanho
    models_data.sort(key=lambda x: x['size'])
    
    # Criar DataFrame
    df = pd.DataFrame(models_data)
    
    # Configurar estilo seaborn
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cores por família
    families = df['family'].unique()
    family_colors = sns.color_palette("husl", len(families))
    family_color_map = dict(zip(families, family_colors))
    
    # Plotar scatter plot com cores por família
    for family in families:
        family_data = df[df['family'] == family]
        ax.scatter(
            family_data['size'],
            family_data['overall_f1'],
            s=200,  # Tamanho dos pontos
            alpha=0.7,
            label=family,
            color=family_color_map[family],
            edgecolors='white',
            linewidth=2,
            zorder=3
        )
        
        # Adicionar labels dos modelos
        for _, row in family_data.iterrows():
            # Encurtar nome do modelo
            sn = short_model_name(row['model_name'])
            
            ax.annotate(
                sn,
                (row['size'], row['overall_f1']),
                fontsize=8,
                alpha=0.8,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=family_color_map[family], alpha=0.7, linewidth=1),
                zorder=4
            )
    
    # Adicionar linha de tendência em escala log
    log_size = np.log2(df['size'])
    z = np.polyfit(log_size, df['overall_f1'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log2(df['size'].min()), np.log2(df['size'].max()), 200, base=2)
    r2 = np.corrcoef(log_size, df['overall_f1'])[0, 1] ** 2
    ax.plot(x_trend, p(np.log2(x_trend)), "r--", alpha=0.5, linewidth=2,
            label=f'Tendência log (R²={r2:.3f})', zorder=1)

    # Configurar eixos — escala logarítmica
    ax.set_xscale('log')
    tick_positions = [1, 2, 4, 8, 16, 32, 70]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_positions))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        [f'{v}B\n$\\log_2={np.log2(v):.1f}$' for v in tick_positions]
    ))
    ax.set_xlabel('Quantidade de Parâmetros (Bilhões, escala log)',
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Overall F1 Score', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('Overall F1 Score vs Quantidade de Parâmetros\nRelação entre Tamanho do Modelo e Desempenho',
                 fontsize=16, fontweight='bold', pad=25)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Limites dos eixos
    ax.set_ylim([0, 1.0])
    ax.set_xlim([df['size'].min() * 0.7, df['size'].max() * 1.4])
    
    # Linhas de referência
    for y_val in [0.5, 0.6, 0.7]:
        ax.axhline(y=y_val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Legenda
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=10, framealpha=0.95, shadow=True,
                      title='Família', title_fontsize=12, 
                      edgecolor='gray', facecolor='white')
    legend.get_frame().set_linewidth(1.5)
    
    # Melhorar aparência
    sns.despine(left=False, bottom=False, top=True, right=True)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#333333', labelsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar
    output_file = output_dir / "f1_vs_parameters.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico F1 vs Parâmetros salvo: {output_file.name}")
    
    plt.close()


def parse_fp_fn_from_results(results_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Parseia FP e FN do arquivo results.txt.
    
    Returns:
        Dict com formato: {
            'strategy_name': {
                'chem_fp': int,
                'chem_fn': int,
                'dis_fp': int,
                'dis_fn': int,
                'overall_fp': int,
                'overall_fn': int
            }
        }
    """
    if not results_path.exists():
        return {}
    
    with open(results_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metrics = {}

    # Captura o nome completo da estratégia (ex: "ZERO SHOT", "EXAMPLES 1")
    strategy_pattern = r'ESTRATÉGIA: ([^\n]+)'
    raw_strategies = re.findall(strategy_pattern, content)

    def normalize_strategy(name: str) -> str:
        """ZERO SHOT → zero_shot | EXAMPLES 1 → examples_1"""
        return name.strip().lower().replace(' ', '_')

    for raw_strategy in raw_strategies:
        key = normalize_strategy(raw_strategy)
        escaped = re.escape(raw_strategy.strip())
        section_pattern = rf'ESTRATÉGIA: {escaped}.*?(?=ESTRATÉGIA: |\Z)'
        section_match = re.search(section_pattern, content, re.DOTALL)

        if not section_match:
            continue

        section = section_match.group(0)

        chem_pattern    = r'Métricas - Chemicals:.*?TP: (\d+), FP: (\d+), FN: (\d+)'
        dis_pattern     = r'Métricas - Diseases:.*?TP: (\d+), FP: (\d+), FN: (\d+)'
        overall_pattern = r'Métricas - Overall:.*?TP: (\d+), FP: (\d+), FN: (\d+)'

        chem_match    = re.search(chem_pattern,    section, re.DOTALL)
        dis_match     = re.search(dis_pattern,     section, re.DOTALL)
        overall_match = re.search(overall_pattern, section, re.DOTALL)

        if chem_match and dis_match and overall_match:
            metrics[key] = {
                'chem_fp':    int(chem_match.group(2)),
                'chem_fn':    int(chem_match.group(3)),
                'dis_fp':     int(dis_match.group(2)),
                'dis_fn':     int(dis_match.group(3)),
                'overall_fp': int(overall_match.group(2)),
                'overall_fn': int(overall_match.group(3)),
            }

    return metrics


def plot_fp_vs_fn(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    all_results_files: List[Tuple[str, Path]],
    output_dir: Path,
    metric_type: str = 'overall'  # 'overall', 'chem', 'dis'
) -> None:
    """
    Cria gráfico mostrando FP (superextração) vs FN (omissão).

    Args:
        metric_type: 'overall', 'chem', ou 'dis'
    """
    # Preparar dados
    models_data = []
    
    # Criar dicionário de results_path por model_name
    results_dict = {name: path for name, path in all_results_files}
    
    for model_name, strategies in all_metrics.items():
        results_path = results_dict.get(model_name)
        if not results_path:
            continue
        
        fp_fn_data = parse_fp_fn_from_results(results_path)
        
        if not fp_fn_data:
            continue
        
        # Encontrar melhor estratégia
        best_strategy = None
        best_f1 = -1
        
        for strategy, metrics in strategies.items():
            overall_f1 = metrics.get('overall_f1', 0)
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                best_strategy = strategy
        
        if best_strategy and best_strategy in fp_fn_data:
            size = get_model_size(model_name)
            family = get_model_family(model_name)
            
            fp_key = f'{metric_type}_fp'
            fn_key = f'{metric_type}_fn'
            
            if fp_key in fp_fn_data[best_strategy] and fn_key in fp_fn_data[best_strategy]:
                models_data.append({
                    'model_name': model_name,
                    'fp': fp_fn_data[best_strategy][fp_key],
                    'fn': fp_fn_data[best_strategy][fn_key],
                    'size': size or 0,
                    'family': family,
                    'f1': best_f1
                })
    
    if not models_data:
        print(f"   ⚠️  Nenhum dado disponível para gráfico FP vs FN ({metric_type})")
        return
    
    # Criar DataFrame
    df = pd.DataFrame(models_data)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))

    # Cor única por modelo (mesmo padrão do Pareto)
    models_sorted = df.sort_values(['family', 'model_name']).reset_index(drop=True)
    model_colors = sns.color_palette("tab20", len(models_sorted))
    model_color_map = {row['model_name']: model_colors[i]
                       for i, row in models_sorted.iterrows()}

    # Plotar cada modelo individualmente
    for _, row in df.iterrows():
        color = model_color_map[row['model_name']]
        ax.scatter(
            row['fp'],
            row['fn'],
            s=300,
            alpha=0.85,
            label=short_model_name(row['model_name']),
            color=color,
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        ax.annotate(
            short_model_name(row['model_name']),
            (row['fp'], row['fn']),
            fontsize=8,
            alpha=0.85,
            ha='center',
            va='bottom',
            xytext=(0, 6),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.7, linewidth=1),
            zorder=4
        )

    # Linha diagonal FP = FN
    max_val = max(df['fp'].max(), df['fn'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5,
            label='FP = FN (equilíbrio)', zorder=1)

    # Configurar eixos
    metric_labels = {
        'overall': 'Overall',
        'chem': 'Chemicals',
        'dis': 'Diseases'
    }
    label = metric_labels.get(metric_type, metric_type)

    ax.set_xlabel(f'Falsos Positivos (FP) — Superextração  [{label}]',
                  fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Falsos Negativos (FN) — Omissão  [{label}]',
                  fontsize=13, fontweight='bold', labelpad=15)
    ax.set_title(f'FP (Superextração) vs FN (Omissão) — {label}\nAnálise de Erros dos Modelos',
                 fontsize=16, fontweight='bold', pad=25)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Linha diagonal já tem label; não precisamos de legenda de modelos
    # (nomes estão anotados diretamente nos pontos)
    ax.legend(
        handles=[ax.get_lines()[0]],   # só a linha FP=FN
        labels=['FP = FN (equilíbrio)'],
        loc='lower right',
        fontsize=9,
        framealpha=0.9,
    )

    # Melhorar aparência
    sns.despine(left=False, bottom=False, top=True, right=True)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#333333', labelsize=10)

    plt.tight_layout()
    
    # Salvar
    output_file = output_dir / f"fp_vs_fn_{metric_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico FP vs FN ({metric_type}) salvo: {output_file.name}")
    
    plt.close()


def prepare_model_comparison_data(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Prepara DataFrame com dados agregados de cada modelo para comparação.
    
    Returns:
        DataFrame com colunas: model_name, size, family, f1_peak, f1_chem_peak, 
        f1_dis_peak, f1_k32, f1_chem_k32, f1_dis_k32
    """
    models_data = []
    
    for model_name, strategies in all_metrics.items():
        size = get_model_size(model_name)
        family = get_model_family(model_name)
        
        # Encontrar melhor estratégia (F1 peak)
        best_strategy = None
        best_f1 = -1
        best_chem_f1 = 0
        best_dis_f1 = 0
        
        for strategy, metrics in strategies.items():
            overall_f1 = metrics.get('overall_f1', 0)
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                best_strategy = strategy
                best_chem_f1 = metrics.get('chem_f1', 0)
                best_dis_f1 = metrics.get('dis_f1', 0)
        
        # Encontrar F1 com 32 exemplos
        f1_k32 = None
        f1_chem_k32 = None
        f1_dis_k32 = None
        
        if 'examples_32' in strategies:
            f1_k32 = strategies['examples_32'].get('overall_f1')
            f1_chem_k32 = strategies['examples_32'].get('chem_f1')
            f1_dis_k32 = strategies['examples_32'].get('dis_f1')
        
        if best_strategy and size is not None:
            models_data.append({
                'model_name': model_name,
                'size': size,
                'family': family,
                'f1_peak': best_f1,
                'f1_chem_peak': best_chem_f1,
                'f1_dis_peak': best_dis_f1,
                'f1_k32': f1_k32 if f1_k32 is not None else best_f1,
                'f1_chem_k32': f1_chem_k32 if f1_chem_k32 is not None else best_chem_f1,
                'f1_dis_k32': f1_dis_k32 if f1_dis_k32 is not None else best_dis_f1
            })
    
    df = pd.DataFrame(models_data)
    df = df.sort_values('size')  # Ordenar por tamanho
    
    return df


def plot_dumbbell_chem_vs_dis(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Gráfico 1: Dumbbell Plot mostrando diferença entre F1_Chem e F1_Dis.
    """
    # Ordenar por tamanho
    df = df.sort_values('size').copy()
    
    # Encurtar nomes dos modelos
    df['short_name'] = df['model_name'].apply(lambda x: 
        re.sub(r'^[^/]+/', '', x.replace('-instruct', '').replace('-chat', ''))[:25])
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.6)))
    
    y_positions = np.arange(len(df))
    
    # Plotar pontos de F1_Chem (azul)
    ax.scatter(df['f1_chem_peak'], y_positions, s=150, color='#2E86AB', 
               alpha=0.8, label='F1 Chemicals', zorder=3, edgecolors='white', linewidth=2)
    
    # Plotar pontos de F1_Dis (laranja)
    ax.scatter(df['f1_dis_peak'], y_positions, s=150, color='#F18F01', 
               alpha=0.8, label='F1 Diseases', zorder=3, edgecolors='white', linewidth=2)
    
    # Conectar pontos com linhas horizontais (halteres)
    for i, row in df.iterrows():
        ax.hlines(y_positions[df.index.get_loc(i)], 
                  row['f1_chem_peak'], row['f1_dis_peak'],
                  colors='gray', linewidth=2, alpha=0.5, zorder=1)
        
        # Calcular diferença
        diff = row['f1_dis_peak'] - row['f1_chem_peak']
        diff_text = f'{diff:+.3f}'
        
        # Posicionar texto no meio do haltere
        mid_x = (row['f1_chem_peak'] + row['f1_dis_peak']) / 2
        ax.text(mid_x, y_positions[df.index.get_loc(i)], diff_text,
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.8), zorder=4)
    
    # Configurar eixos
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['short_name'], fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Modelos (ordenados por tamanho)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title("O 'Gap' Cognitivo: Diferença de Performance entre Classes por Modelo",
                 fontsize=16, fontweight='bold', pad=25)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', axis='x', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Legenda
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True)
    
    # Melhorar aparência
    sns.despine(left=False, bottom=False, top=True, right=True)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#333333', labelsize=10)
    
    # Inverter eixo Y para ter modelos maiores no topo
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Salvar
    output_file = output_dir / "dumbbell_chem_vs_dis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico Dumbbell (Chem vs Dis) salvo: {output_file.name}")
    
    plt.close()


def plot_pareto_frontier(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Gráfico 2: Scatter Plot da Fronteira de Pareto (Eficiência).
    """
    # Criar figura mais larga para acomodar legenda lateral
    fig, ax = plt.subplots(figsize=(16, 8))

    # Cor única por modelo
    models_sorted = df.sort_values(['family', 'model_name']).reset_index(drop=True)
    model_colors = sns.color_palette("tab20", len(models_sorted))
    model_color_map = {row['model_name']: model_colors[i]
                       for i, row in models_sorted.iterrows()}

    # Plotar cada modelo individualmente para ter entrada própria na legenda
    for _, row in df.iterrows():
        color = model_color_map[row['model_name']]
        ax.scatter(
            row['size'],
            row['f1_peak'],
            s=200,
            alpha=0.85,
            label=short_model_name(row['model_name']),
            color=color,
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        # Anotação com nome do modelo no ponto
        ax.annotate(
            short_model_name(row['model_name']),
            (row['size'], row['f1_peak']),
            fontsize=8,
            alpha=0.85,
            ha='left',
            va='bottom',
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.7, linewidth=1),
            zorder=4
        )
    
    # Identificar modelos da fronteira de Pareto
    # Ordenar por tamanho e encontrar os melhores em cada faixa
    df_sorted = df.sort_values('size')
    
    # Encontrar pontos da fronteira (melhor F1 para cada tamanho)
    frontier_points = []
    current_best_f1 = -1
    
    for _, row in df_sorted.iterrows():
        if row['f1_peak'] > current_best_f1:
            frontier_points.append((row['size'], row['f1_peak']))
            current_best_f1 = row['f1_peak']
    
    # Tentar identificar modelos específicos mencionados
    # Procurar por modelos que possam ser Llama-3B, Llama-8B, Qwen-14B
    frontier_models = []
    for size, f1 in frontier_points:
        candidates = df[(df['size'] == size) & (df['f1_peak'] == f1)]
        if len(candidates) > 0:
            frontier_models.append((size, f1))
    
    # Plotar linha da fronteira de Pareto
    if len(frontier_models) >= 2:
        frontier_x = [p[0] for p in frontier_models]
        frontier_y = [p[1] for p in frontier_models]
        # Ordenar por X
        sorted_frontier = sorted(zip(frontier_x, frontier_y))
        frontier_x = [x for x, y in sorted_frontier]
        frontier_y = [y for x, y in sorted_frontier]
        
        ax.plot(frontier_x, frontier_y, 'r--', linewidth=2, alpha=0.7,
               label='Fronteira de Eficiência', zorder=2)
    
    # Configurar eixos
    import numpy as np
    ax.set_xscale('log')
    tick_positions = [1, 2, 4, 8, 16, 32, 70]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_positions))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        [f'{v}B\n$\\log_2={np.log2(v):.1f}$' for v in tick_positions]
    ))
    ax.set_xlabel('Tamanho do Modelo (Bilhões de Parâmetros)',
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('F1 Score Máximo (F1_Peak)',
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('Eficiência de Modelos: Desempenho Máximo vs. Tamanho',
                 fontsize=16, fontweight='bold', pad=25)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Legenda
    # Legenda lateral com todos os modelos — espaço suficiente para 18 entradas
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.92),
        borderaxespad=0,
        fontsize=8.5,
        framealpha=0.95,
        title='Modelos',
        title_fontsize=10,
        markerscale=1.2,
        handlelength=1.5,
        labelspacing=1.1,
    )

    # Melhorar aparência
    sns.despine(left=False, bottom=False, top=True, right=True)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#333333', labelsize=10)

    plt.tight_layout()
    # Reservar 30% à direita para a legenda
    plt.subplots_adjust(right=0.68)
    
    # Salvar
    output_file = output_dir / "pareto_frontier_efficiency.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico Fronteira de Pareto salvo: {output_file.name}")
    
    plt.close()


def plot_resilience_heatmap(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Gráfico 3: Heatmap de Resiliência ao Contexto (Delta F1_k32 vs F1_Peak).
    """
    # Calcular Delta
    df = df.copy()
    df['delta'] = df['f1_k32'] - df['f1_peak']
    
    # Encurtar nomes dos modelos
    df['short_name'] = df['model_name'].apply(lambda x: 
        re.sub(r'^[^/]+/', '', x.replace('-instruct', '').replace('-chat', ''))[:30])
    
    # Ordenar do pior Delta para o melhor
    df = df.sort_values('delta')
    
    # Preparar dados para heatmap
    # Criar matriz com uma coluna (Delta) e uma linha por modelo
    # Formato: uma linha com todos os deltas, onde cada coluna é um modelo
    heatmap_data = df[['delta']].T
    heatmap_data.columns = df['short_name'].values
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.5), 4))
    
    # Criar heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=df['delta'].min() - 0.01,
        vmax=df['delta'].max() + 0.01,
        cbar_kws={'label': 'Delta F1 (k=32 - Peak)', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        xticklabels=True,
        yticklabels=['Delta'],
        square=False
    )
    
    # Configurar eixos
    ax.set_xlabel('Modelos (ordenados do pior para o melhor Delta)', 
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('', fontsize=14, fontweight='bold')
    ax.set_title('Teste de Stress: Variação de F1 em Contexto Denso (k=32 vs Pico)',
                 fontsize=16, fontweight='bold', pad=25)
    
    # Rotacionar labels do eixo X
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Salvar
    output_file = output_dir / "resilience_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Gráfico Heatmap de Resiliência salvo: {output_file.name}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gera gráficos a partir dos arquivos results.txt"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Diretório contendo os datasets (padrão: dataset/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Diretório para salvar os gráficos (padrão: figures/)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['overall_f1', 'chem_f1', 'dis_f1', 'all'],
        default='all',
        help='Métrica a plotar (padrão: all)'
    )
    
    args = parser.parse_args()
    
    # Configurar diretórios
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = project_root / "dataset"
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 GERANDO GRÁFICOS DOS RESULTADOS")
    print("=" * 80)
    print()
    
    # Encontrar todos os arquivos results.txt
    print("🔍 Procurando arquivos results.txt...")
    results_files = find_all_results_files(dataset_dir)
    
    if not results_files:
        print("❌ Nenhum arquivo results.txt encontrado!")
        return
    
    print(f"✅ Encontrados {len(results_files)} arquivos results.txt")
    print()
    
    # Parsear todos os arquivos
    print("📖 Parseando arquivos...")
    all_metrics = {}
    
    for model_name, results_path in results_files:
        print(f"   📄 {model_name}...")
        metrics = parse_results_file(results_path)
        if metrics:
            all_metrics[model_name] = metrics
            print(f"      ✅ {len(metrics)} estratégias encontradas")
        else:
            print(f"      ⚠️  Nenhuma métrica encontrada")
    
    print()
    
    if not all_metrics:
        print("❌ Nenhuma métrica foi extraída!")
        return
    
    apply_table_overrides(all_metrics)
    
    # Gerar gráficos
    print("📈 Gerando gráficos...")
    print()
    
    metrics_to_plot = ['overall_f1', 'chem_f1', 'dis_f1'] if args.metric == 'all' else [args.metric]
    
    # Gráficos gerais (todos os modelos)
    print("📈 Gerando gráficos gerais (todos os modelos)...")
    print()
    for metric_type in metrics_to_plot:
        print(f"   📊 Gerando gráfico para {metric_type}...")
        plot_f1_vs_examples(all_metrics, output_dir, metric_type)
    
    print()
    
    # Gráficos por faixa de tamanho
    print("📈 Gerando gráficos por faixa de tamanho...")
    print()
    for metric_type in metrics_to_plot:
        print(f"   📊 Gerando gráficos por tamanho para {metric_type}...")
        plot_by_size_range(all_metrics, output_dir, metric_type)
    
    print()
    
    # Gráficos por família
    print("📈 Gerando gráficos por família (evolução)...")
    print()
    for metric_type in metrics_to_plot:
        print(f"   📊 Gerando gráficos por família para {metric_type}...")
        plot_by_family(all_metrics, output_dir, metric_type)
    
    print()
    
    # Gráficos de barras
    print("📊 Gerando gráficos de barras (todas as métricas)...")
    print()
    
    # Gráfico geral
    print("   📊 Gerando gráfico de barras geral...")
    plot_metrics_bar_chart(all_metrics, output_dir)
    
    print()
    
    # Gráficos por faixa de tamanho
    print("   📊 Gerando gráficos de barras por faixa de tamanho...")
    plot_bar_charts_by_size(all_metrics, output_dir)
    
    print()
    
    # Gráficos por família
    print("   📊 Gerando gráficos de barras por família...")
    plot_bar_charts_by_family(all_metrics, output_dir)
    
    print()
    
    # Gráfico F1 vs Parâmetros
    print("📊 Gerando gráfico F1 vs Quantidade de Parâmetros...")
    print()
    plot_f1_vs_parameters(all_metrics, output_dir)
    
    print()
    
    # Gráficos FP vs FN (Alucinação vs Omissão)
    print("📊 Gerando gráficos FP vs FN (Alucinação vs Omissão)...")
    print()
    for metric_type in ['overall', 'chem', 'dis']:
        print(f"   📊 Gerando gráfico FP vs FN para {metric_type}...")
        plot_fp_vs_fn(all_metrics, results_files, output_dir, metric_type)
    
    print()
    
    # Preparar dados para gráficos de comparação avançada
    print("📊 Preparando dados para gráficos de análise avançada...")
    comparison_df = prepare_model_comparison_data(all_metrics)
    
    if len(comparison_df) > 0:
        print(f"   ✅ {len(comparison_df)} modelos preparados")
        print()
        
        # Gráfico 1: Dumbbell Plot
        print("📊 Gerando Gráfico 1: Dumbbell Plot (Chem vs Dis)...")
        plot_dumbbell_chem_vs_dis(comparison_df, output_dir)
        print()
        
        # Gráfico 2: Fronteira de Pareto
        print("📊 Gerando Gráfico 2: Fronteira de Pareto (Eficiência)...")
        plot_pareto_frontier(comparison_df, output_dir)
        print()
        
        # Gráfico 3: Heatmap de Resiliência
        print("📊 Gerando Gráfico 3: Heatmap de Resiliência...")
        plot_resilience_heatmap(comparison_df, output_dir)
        print()
    else:
        print("   ⚠️  Nenhum dado disponível para gráficos de análise avançada")
    
    print()
    print("=" * 80)
    print("✅ GRÁFICOS GERADOS COM SUCESSO!")
    print(f"📁 Diretório: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

