#!/usr/bin/env python3
"""
Script consolidado para gerar arquivos de comparison, inferencias e results.txt.
Pode processar um modelo específico ou todos os modelos.

Uso:
    python pipeline/get_results.py [--model MODEL_NAME] [--prompt-type TYPE] [--strategy STRATEGY]
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import List, Dict, Optional

# Adicionar paths
sys.path.insert(0, str(Path(__file__).parent))                               # pipeline/
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "audit"))

from indicios_to_df import parse_indicios_to_df

# Estratégias padrão
DEFAULT_STRATEGIES = ["zero_shot", "examples_1", "examples_2", "examples_4", "examples_8", "examples_16", "examples_32"]

K_MAP = {'zero_shot': 0, 'examples_1': 1, 'examples_2': 2, 'examples_4': 4, 'examples_8': 8, 'examples_16': 16, 'examples_32': 32}

def normalize_text(text: str) -> str:
    """Normaliza texto para comparação"""
    if pd.isna(text) or not text:
        return ""
    return str(text).lower().strip()

def parse_entities_list(entity_str: str) -> List[str]:
    """Parseia lista de entidades"""
    if pd.isna(entity_str) or not entity_str:
        return []
    try:
        if isinstance(entity_str, str):
            parsed = json.loads(entity_str)
            if isinstance(parsed, list):
                return [str(e).lower().strip() for e in parsed]
        return []
    except:
        return []

def calculate_metrics_for_article(gold_chem: List[str], pred_chem: List[str], gold_dis: List[str], pred_dis: List[str]):
    """Calcula métricas para um artigo"""
    # Chemicals
    gold_chem_set = set(gold_chem)
    pred_chem_set = set(pred_chem)
    chem_tp = len(gold_chem_set & pred_chem_set)
    chem_fp = len(pred_chem_set - gold_chem_set)
    chem_fn = len(gold_chem_set - pred_chem_set)
    chem_precision = chem_tp / (chem_tp + chem_fp) if (chem_tp + chem_fp) > 0 else 0.0
    chem_recall = chem_tp / (chem_tp + chem_fn) if (chem_tp + chem_fn) > 0 else 0.0
    chem_f1 = 2 * (chem_precision * chem_recall) / (chem_precision + chem_recall) if (chem_precision + chem_recall) > 0 else 0.0
    
    # Diseases
    gold_dis_set = set(gold_dis)
    pred_dis_set = set(pred_dis)
    dis_tp = len(gold_dis_set & pred_dis_set)
    dis_fp = len(pred_dis_set - gold_dis_set)
    dis_fn = len(gold_dis_set - pred_dis_set)
    dis_precision = dis_tp / (dis_tp + dis_fp) if (dis_tp + dis_fp) > 0 else 0.0
    dis_recall = dis_tp / (dis_tp + dis_fn) if (dis_tp + dis_fn) > 0 else 0.0
    dis_f1 = 2 * (dis_precision * dis_recall) / (dis_precision + dis_recall) if (dis_precision + dis_recall) > 0 else 0.0
    
    # Overall
    overall_tp = chem_tp + dis_tp
    overall_fp = chem_fp + dis_fp
    overall_fn = chem_fn + dis_fn
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        'chem_precision': chem_precision, 'chem_recall': chem_recall, 'chem_f1': chem_f1,
        'chem_tp': chem_tp, 'chem_fp': chem_fp, 'chem_fn': chem_fn,
        'dis_precision': dis_precision, 'dis_recall': dis_recall, 'dis_f1': dis_f1,
        'dis_tp': dis_tp, 'dis_fp': dis_fp, 'dis_fn': dis_fn,
        'overall_precision': overall_precision, 'overall_recall': overall_recall, 'overall_f1': overall_f1,
        'overall_tp': overall_tp, 'overall_fp': overall_fp, 'overall_fn': overall_fn,
        'chem_tp_list': list(gold_chem_set & pred_chem_set),
        'chem_fp_list': list(pred_chem_set - gold_chem_set),
        'chem_fn_list': list(gold_chem_set - pred_chem_set),
        'dis_tp_list': list(gold_dis_set & pred_dis_set),
        'dis_fp_list': list(pred_dis_set - gold_dis_set),
        'dis_fn_list': list(gold_dis_set - pred_dis_set),
    }

def create_comparison_csv(gold_csv: str, pred_csv: str, output_csv: str) -> pd.DataFrame:
    """Cria CSV de comparação"""
    print(f"📊 Comparando: {Path(pred_csv).name} vs Gold")
    
    df_gold = pd.read_csv(gold_csv)
    df_pred = pd.read_csv(pred_csv)
    
    # Renomear colunas para evitar conflito no merge
    df_gold_renamed = df_gold.rename(columns={
        'text': 'text_gold',
        'chemicals_list': 'chemicals_list_gold',
        'diseases_list': 'diseases_list_gold'
    })
    df_pred_renamed = df_pred.rename(columns={
        'text': 'text_pred',
        'chemicals_list': 'chemicals_list_pred',
        'diseases_list': 'diseases_list_pred'
    })
    
    # Merge por PMID
    df_merged = df_gold_renamed.merge(df_pred_renamed, on='pmid', how='outer')
    
    comparison_rows = []
    for _, row in df_merged.iterrows():
        present_in = 'both' if pd.notna(row.get('text_gold')) and pd.notna(row.get('text_pred')) else ('gold' if pd.notna(row.get('text_gold')) else 'pred')
        
        text_gold = row.get('text_gold', '')
        text_pred = row.get('text_pred', '')
        
        gold_chem = parse_entities_list(row.get('chemicals_list_gold', '[]'))
        pred_chem = parse_entities_list(row.get('chemicals_list_pred', '[]'))
        gold_dis = parse_entities_list(row.get('diseases_list_gold', '[]'))
        pred_dis = parse_entities_list(row.get('diseases_list_pred', '[]'))
        
        metrics = calculate_metrics_for_article(gold_chem, pred_chem, gold_dis, pred_dis) if present_in == 'both' else {
            'chem_precision': 0.0, 'chem_recall': 0.0, 'chem_f1': 0.0, 'chem_tp': 0, 'chem_fp': 0, 'chem_fn': 0,
            'dis_precision': 0.0, 'dis_recall': 0.0, 'dis_f1': 0.0, 'dis_tp': 0, 'dis_fp': 0, 'dis_fn': 0,
            'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0,
            'chem_tp_list': [], 'chem_fp_list': [], 'chem_fn_list': [],
            'dis_tp_list': [], 'dis_fp_list': [], 'dis_fn_list': [],
        }
        
        comp_row = {
            'pmid': row['pmid'],
            'text_gold': text_gold,
            'text_pred': text_pred,
            'present_in': present_in,
            'gold_chemicals_all': json.dumps(gold_chem),
            'pred_chemicals_all': json.dumps(pred_chem),
            'gold_diseases_all': json.dumps(gold_dis),
            'pred_diseases_all': json.dumps(pred_dis),
            'gold_num_chemicals': len(gold_chem),
            'gold_num_diseases': len(gold_dis),
            'gold_num_entities': len(gold_chem) + len(gold_dis),
            'pred_num_chemicals': len(pred_chem),
            'pred_num_diseases': len(pred_dis),
            'pred_num_entities': len(pred_chem) + len(pred_dis),
            'chem_precision': metrics['chem_precision'],
            'chem_recall': metrics['chem_recall'],
            'chem_f1': metrics['chem_f1'],
            'chem_tp': metrics['chem_tp'],
            'chem_fp': metrics['chem_fp'],
            'chem_fn': metrics['chem_fn'],
            'dis_precision': metrics['dis_precision'],
            'dis_recall': metrics['dis_recall'],
            'dis_f1': metrics['dis_f1'],
            'dis_tp': metrics['dis_tp'],
            'dis_fp': metrics['dis_fp'],
            'dis_fn': metrics['dis_fn'],
            'overall_precision': metrics['overall_precision'],
            'overall_recall': metrics['overall_recall'],
            'overall_f1': metrics['overall_f1'],
            'overall_tp': metrics['overall_tp'],
            'overall_fp': metrics['overall_fp'],
            'overall_fn': metrics['overall_fn'],
            'chem_tp_list': json.dumps(metrics['chem_tp_list']),
            'chem_fp_list': json.dumps(metrics['chem_fp_list']),
            'chem_fn_list': json.dumps(metrics['chem_fn_list']),
            'dis_tp_list': json.dumps(metrics['dis_tp_list']),
            'dis_fp_list': json.dumps(metrics['dis_fp_list']),
            'dis_fn_list': json.dumps(metrics['dis_fn_list']),
        }
        comparison_rows.append(comp_row)
    
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(output_csv, index=False)
    print(f"💾 Comparação salva: {output_csv}")
    return df_comp

def process_strategy(strategy: str, strategy_dir: Path, gold_csv: Path, output_base: Path, model_name: str, prompt_type: str) -> Optional[Dict]:
    """Processa uma estratégia"""
    print(f"  📝 {strategy}")
    
    inferencias_dir = output_base / "inferencias"
    comparison_dir = output_base / "comparison"
    inferencias_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Converter JSONs
    inferencias_csv = inferencias_dir / f"{model_name}_{strategy}.csv"
    try:
        df_pred = parse_indicios_to_df(str(strategy_dir), output_path=str(inferencias_csv))
        if len(df_pred) == 0:
            print(f"    ⚠️  Nenhum artigo")
            return None
    except Exception as e:
        print(f"    ❌ Erro: {e}")
        return None
    
    # 2. Comparar
    comparison_csv = comparison_dir / f"comparison_{strategy}.csv"
    try:
        df_comp = create_comparison_csv(str(gold_csv), str(inferencias_csv), str(comparison_csv))
        df_both = df_comp[df_comp['present_in'] == 'both']
        if len(df_both) == 0:
            return None
        
        # Calcular métricas agregadas
        return {
            'strategy': strategy,
            'num_articles': len(df_pred),
            'gold_chem': df_both['gold_num_chemicals'].sum(),
            'gold_dis': df_both['gold_num_diseases'].sum(),
            'pred_chem': df_both['pred_num_chemicals'].sum(),
            'pred_dis': df_both['pred_num_diseases'].sum(),
            'chem_precision_macro': df_both['chem_precision'].mean(),
            'chem_recall_macro': df_both['chem_recall'].mean(),
            'chem_f1_macro': df_both['chem_f1'].mean(),
            'chem_precision_median': df_both['chem_precision'].median(),
            'chem_recall_median': df_both['chem_recall'].median(),
            'chem_f1_median': df_both['chem_f1'].median(),
            'chem_precision_std': df_both['chem_precision'].std(),
            'chem_recall_std': df_both['chem_recall'].std(),
            'chem_f1_std': df_both['chem_f1'].std(),
            'chem_precision_micro': df_both['chem_tp'].sum() / (df_both['chem_tp'].sum() + df_both['chem_fp'].sum()) if (df_both['chem_tp'].sum() + df_both['chem_fp'].sum()) > 0 else 0.0,
            'chem_recall_micro': df_both['chem_tp'].sum() / (df_both['chem_tp'].sum() + df_both['chem_fn'].sum()) if (df_both['chem_tp'].sum() + df_both['chem_fn'].sum()) > 0 else 0.0,
            'chem_f1_micro': 0.0,  # Calculado depois
            'chem_tp': df_both['chem_tp'].sum(),
            'chem_fp': df_both['chem_fp'].sum(),
            'chem_fn': df_both['chem_fn'].sum(),
            'dis_precision_macro': df_both['dis_precision'].mean(),
            'dis_recall_macro': df_both['dis_recall'].mean(),
            'dis_f1_macro': df_both['dis_f1'].mean(),
            'dis_precision_median': df_both['dis_precision'].median(),
            'dis_recall_median': df_both['dis_recall'].median(),
            'dis_f1_median': df_both['dis_f1'].median(),
            'dis_precision_std': df_both['dis_precision'].std(),
            'dis_recall_std': df_both['dis_recall'].std(),
            'dis_f1_std': df_both['dis_f1'].std(),
            'dis_precision_micro': df_both['dis_tp'].sum() / (df_both['dis_tp'].sum() + df_both['dis_fp'].sum()) if (df_both['dis_tp'].sum() + df_both['dis_fp'].sum()) > 0 else 0.0,
            'dis_recall_micro': df_both['dis_tp'].sum() / (df_both['dis_tp'].sum() + df_both['dis_fn'].sum()) if (df_both['dis_tp'].sum() + df_both['dis_fn'].sum()) > 0 else 0.0,
            'dis_f1_micro': 0.0,
            'dis_tp': df_both['dis_tp'].sum(),
            'dis_fp': df_both['dis_fp'].sum(),
            'dis_fn': df_both['dis_fn'].sum(),
            'overall_precision_macro': df_both['overall_precision'].mean(),
            'overall_recall_macro': df_both['overall_recall'].mean(),
            'overall_f1_macro': df_both['overall_f1'].mean(),
            'overall_precision_median': df_both['overall_precision'].median(),
            'overall_recall_median': df_both['overall_recall'].median(),
            'overall_f1_median': df_both['overall_f1'].median(),
            'overall_precision_std': df_both['overall_precision'].std(),
            'overall_recall_std': df_both['overall_recall'].std(),
            'overall_f1_std': df_both['overall_f1'].std(),
            'overall_precision_micro': df_both['overall_tp'].sum() / (df_both['overall_tp'].sum() + df_both['overall_fp'].sum()) if (df_both['overall_tp'].sum() + df_both['overall_fp'].sum()) > 0 else 0.0,
            'overall_recall_micro': df_both['overall_tp'].sum() / (df_both['overall_tp'].sum() + df_both['overall_fn'].sum()) if (df_both['overall_tp'].sum() + df_both['overall_fn'].sum()) > 0 else 0.0,
            'overall_f1_micro': 0.0,
            'overall_tp': df_both['overall_tp'].sum(),
            'overall_fp': df_both['overall_fp'].sum(),
            'overall_fn': df_both['overall_fn'].sum(),
            'df_comp': df_comp,  # Guardar para análise posterior
        }
    except Exception as e:
        print(f"    ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_results_txt(all_results: List[Dict], output_base: Path, model_name: str, prompt_type: str):
    """Gera results.txt no formato completo e detalhado"""
    if not all_results:
        return
    
    results_file = output_base / "results.txt"
    
    # Preparar dados
    results_data = []
    for r in all_results:
        # Calcular F1 macro
        chem_f1_micro = 2 * (r['chem_precision_micro'] * r['chem_recall_micro']) / (r['chem_precision_micro'] + r['chem_recall_micro']) if (r['chem_precision_micro'] + r['chem_recall_micro']) > 0 else 0.0
        dis_f1_micro = 2 * (r['dis_precision_micro'] * r['dis_recall_micro']) / (r['dis_precision_micro'] + r['dis_recall_micro']) if (r['dis_precision_micro'] + r['dis_recall_micro']) > 0 else 0.0
        overall_f1_micro = 2 * (r['overall_precision_micro'] * r['overall_recall_micro']) / (r['overall_precision_micro'] + r['overall_recall_micro']) if (r['overall_precision_micro'] + r['overall_recall_micro']) > 0 else 0.0
        r['chem_f1_micro'] = chem_f1_micro
        r['dis_f1_micro'] = dis_f1_micro
        r['overall_f1_micro'] = overall_f1_micro
        results_data.append(r)
    
    df_results = pd.DataFrame(results_data)
    df_results['k'] = df_results['strategy'].map(K_MAP)
    df_results = df_results.sort_values('k')
    
    # Calcular estatísticas finais
    total_articles = df_results['num_articles'].sum()
    total_gold_chem = df_results['gold_chem'].sum()
    total_gold_dis = df_results['gold_dis'].sum()
    total_gold = total_gold_chem + total_gold_dis
    total_pred_chem = df_results['pred_chem'].sum()
    total_pred_dis = df_results['pred_dis'].sum()
    total_pred = total_pred_chem + total_pred_dis
    
    with open(results_file, 'w', encoding='utf-8') as f:
        # Cabeçalho
        f.write("="*100 + "\n")
        f.write("ANÁLISE DETALHADA - COMPARAÇÃO DE ESTRATÉGIAS DE PROMPT\n")
        f.write("="*100 + "\n")
        f.write(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Tipo de Prompt: {prompt_type}\n\n")
        
        # RESUMO EXECUTIVO
        f.write("="*100 + "\n")
        f.write("RESUMO EXECUTIVO\n")
        f.write("="*100 + "\n\n")
        f.write("Métricas Gerais por Estratégia (Macro-Average)\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Estratégia':<15} {'Artigos':<10} {'Chem P':<10} {'Chem R':<10} {'Chem F1':<10} {'Dis P':<10} {'Dis R':<10} {'Dis F1':<10} {'Overall F1':<12}\n")
        f.write("-"*100 + "\n")
        for _, row in df_results.iterrows():
            f.write(f"{row['strategy']:<15} {row['num_articles']:<10} {row['chem_precision_macro']:<10.4f} {row['chem_recall_macro']:<10.4f} {row['chem_f1_macro']:<10.4f} {row['dis_precision_macro']:<10.4f} {row['dis_recall_macro']:<10.4f} {row['dis_f1_macro']:<10.4f} {row['overall_f1_macro']:<12.4f}\n")
        
        f.write("\nMétricas Micro (Agregado)\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Estratégia':<15} {'Chem P':<10} {'Chem R':<10} {'Chem F1':<10} {'Dis P':<10} {'Dis R':<10} {'Dis F1':<10} {'Overall F1':<12}\n")
        f.write("-"*100 + "\n")
        for _, row in df_results.iterrows():
            f.write(f"{row['strategy']:<15} {row['chem_precision_micro']:<10.4f} {row['chem_recall_micro']:<10.4f} {row['chem_f1_micro']:<10.4f} {row['dis_precision_micro']:<10.4f} {row['dis_recall_micro']:<10.4f} {row['dis_f1_micro']:<10.4f} {row['overall_f1_micro']:<12.4f}\n")
        
        # ANÁLISE DETALHADA POR ESTRATÉGIA
        f.write("\n" + "="*100 + "\n")
        f.write("ANÁLISE DETALHADA POR ESTRATÉGIA\n")
        f.write("="*100 + "\n\n")
        
        for _, row in df_results.iterrows():
            strategy_name = row['strategy'].upper().replace('_', ' ')
            f.write("-"*100 + "\n")
            f.write(f"ESTRATÉGIA: {strategy_name}\n")
            f.write("-"*100 + "\n\n")
            
            f.write("Informações Básicas:\n")
            f.write(f"  Total de artigos: {int(row['num_articles'])}\n")
            f.write(f"  Total de entidades Gold: {int(row['gold_chem'] + row['gold_dis'])} (Chemicals: {int(row['gold_chem'])}, Diseases: {int(row['gold_dis'])})\n")
            f.write(f"  Total de entidades Preditas: {int(row['pred_chem'] + row['pred_dis'])} (Chemicals: {int(row['pred_chem'])}, Diseases: {int(row['pred_dis'])})\n\n")
            
            # Chemicals
            f.write("Métricas - Chemicals:\n")
            f.write(f"  Precision (Macro): {row['chem_precision_macro']:.4f} (Mediana: {row['chem_precision_median']:.4f}, Std: {row['chem_precision_std']:.4f})\n")
            f.write(f"  Recall (Macro):    {row['chem_recall_macro']:.4f} (Mediana: {row['chem_recall_median']:.4f}, Std: {row['chem_recall_std']:.4f})\n")
            f.write(f"  F1 (Macro):         {row['chem_f1_macro']:.4f} (Mediana: {row['chem_f1_median']:.4f}, Std: {row['chem_f1_std']:.4f})\n")
            f.write(f"  Precision (Micro):       {row['chem_precision_micro']:.4f}\n")
            f.write(f"  Recall (Micro):          {row['chem_recall_micro']:.4f}\n")
            f.write(f"  F1 (Micro):              {row['chem_f1_micro']:.4f}\n")
            f.write(f"  TP: {int(row['chem_tp'])}, FP: {int(row['chem_fp'])}, FN: {int(row['chem_fn'])}\n\n")

            # Diseases
            f.write("Métricas - Diseases:\n")
            f.write(f"  Precision (Macro): {row['dis_precision_macro']:.4f} (Mediana: {row['dis_precision_median']:.4f}, Std: {row['dis_precision_std']:.4f})\n")
            f.write(f"  Recall (Macro):    {row['dis_recall_macro']:.4f} (Mediana: {row['dis_recall_median']:.4f}, Std: {row['dis_recall_std']:.4f})\n")
            f.write(f"  F1 (Macro):         {row['dis_f1_macro']:.4f} (Mediana: {row['dis_f1_median']:.4f}, Std: {row['dis_f1_std']:.4f})\n")
            f.write(f"  Precision (Micro):       {row['dis_precision_micro']:.4f}\n")
            f.write(f"  Recall (Micro):          {row['dis_recall_micro']:.4f}\n")
            f.write(f"  F1 (Micro):              {row['dis_f1_micro']:.4f}\n")
            f.write(f"  TP: {int(row['dis_tp'])}, FP: {int(row['dis_fp'])}, FN: {int(row['dis_fn'])}\n\n")

            # Overall
            f.write("Métricas - Overall:\n")
            f.write(f"  Precision (Macro): {row['overall_precision_macro']:.4f} (Mediana: {row['overall_precision_median']:.4f}, Std: {row['overall_precision_std']:.4f})\n")
            f.write(f"  Recall (Macro):    {row['overall_recall_macro']:.4f} (Mediana: {row['overall_recall_median']:.4f}, Std: {row['overall_recall_std']:.4f})\n")
            f.write(f"  F1 (Macro):         {row['overall_f1_macro']:.4f} (Mediana: {row['overall_f1_median']:.4f}, Std: {row['overall_f1_std']:.4f})\n")
            f.write(f"  Precision (Micro):       {row['overall_precision_micro']:.4f}\n")
            f.write(f"  Recall (Micro):          {row['overall_recall_micro']:.4f}\n")
            f.write(f"  F1 (Micro):              {row['overall_f1_micro']:.4f}\n")
            f.write(f"  TP: {int(row['overall_tp'])}, FP: {int(row['overall_fp'])}, FN: {int(row['overall_fn'])}\n\n")
        
        # COMPARAÇÃO ENTRE ESTRATÉGIAS
        f.write("\n" + "="*100 + "\n")
        f.write("COMPARAÇÃO ENTRE ESTRATÉGIAS\n")
        f.write("="*100 + "\n\n")
        
        f.write("Melhor Estratégia por Métrica:\n")
        f.write("-"*100 + "\n")
        best_chem_f1_macro = df_results.loc[df_results['chem_f1_macro'].idxmax()]
        best_chem_f1_micro = df_results.loc[df_results['chem_f1_micro'].idxmax()]
        best_dis_f1_macro = df_results.loc[df_results['dis_f1_macro'].idxmax()]
        best_dis_f1_micro = df_results.loc[df_results['dis_f1_micro'].idxmax()]
        best_overall_f1_macro = df_results.loc[df_results['overall_f1_macro'].idxmax()]
        best_overall_f1_micro = df_results.loc[df_results['overall_f1_micro'].idxmax()]
        
        f.write(f"  Chemicals F1 (Macro)          : {best_chem_f1_macro['strategy']:<15} ({best_chem_f1_macro['chem_f1_macro']:.4f})\n")
        f.write(f"  Chemicals F1 (Micro)          : {best_chem_f1_micro['strategy']:<15} ({best_chem_f1_micro['chem_f1_micro']:.4f})\n")
        f.write(f"  Diseases F1 (Macro)           : {best_dis_f1_macro['strategy']:<15} ({best_dis_f1_macro['dis_f1_macro']:.4f})\n")
        f.write(f"  Diseases F1 (Micro)           : {best_dis_f1_micro['strategy']:<15} ({best_dis_f1_micro['dis_f1_micro']:.4f})\n")
        f.write(f"  Overall F1 (Macro)            : {best_overall_f1_macro['strategy']:<15} ({best_overall_f1_macro['overall_f1_macro']:.4f})\n")
        f.write(f"  Overall F1 (Micro)            : {best_overall_f1_micro['strategy']:<15} ({best_overall_f1_micro['overall_f1_micro']:.4f})\n\n")
        
        f.write("Análise de Tendências:\n")
        f.write("-"*100 + "\n")
        f.write("Evolução do Overall F1 (Micro/Agregado) por número de exemplos:\n")
        max_f1 = df_results['overall_f1_micro'].max()
        for _, row in df_results.iterrows():
            k = row['k']
            strategy = row['strategy']
            f1 = row['overall_f1_micro']
            bar_length = int((f1 / max_f1) * 20) if max_f1 > 0 else 0
            bar = "█" * bar_length
            f.write(f"  {strategy:<15} ({k:>2} exemplos): {f1:.4f} {bar}\n")
        
        # ANÁLISE DE FP, FN E TP
        f.write("\n" + "="*100 + "\n")
        f.write("ANÁLISE DETALHADA DE FP, FN E TP\n")
        f.write("="*100 + "\n\n")
        
        f.write("FALSOS POSITIVOS (FP) - Análise por Tipo:\n")
        f.write("-"*100 + "\n")
        chem_fp_max = df_results.loc[df_results['chem_fp'].idxmax()]
        chem_fp_min = df_results.loc[df_results['chem_fp'].idxmin()]
        dis_fp_max = df_results.loc[df_results['dis_fp'].idxmax()]
        dis_fp_min = df_results.loc[df_results['dis_fp'].idxmin()]
        avg_chem_fp = df_results['chem_fp'].mean()
        avg_dis_fp = df_results['dis_fp'].mean()
        total_chem_fp = df_results['chem_fp'].sum()
        total_dis_fp = df_results['dis_fp'].sum()
        
        f.write(f"  Chemicals FP:\n")
        f.write(f"    Maior: {chem_fp_max['strategy']} com {int(chem_fp_max['chem_fp'])} FP ({chem_fp_max['chem_fp']/total_chem_fp*100:.2f}% do total)\n")
        f.write(f"    Menor: {chem_fp_min['strategy']} com {int(chem_fp_min['chem_fp'])} FP ({chem_fp_min['chem_fp']/total_chem_fp*100:.2f}% do total)\n\n")
        f.write(f"  Diseases FP:\n")
        f.write(f"    Maior: {dis_fp_max['strategy']} com {int(dis_fp_max['dis_fp'])} FP ({dis_fp_max['dis_fp']/total_dis_fp*100:.2f}% do total)\n")
        f.write(f"    Menor: {dis_fp_min['strategy']} com {int(dis_fp_min['dis_fp'])} FP ({dis_fp_min['dis_fp']/total_dis_fp*100:.2f}% do total)\n\n")
        
        if avg_chem_fp > avg_dis_fp:
            f.write(f"  ⚠️  Chemicals geram MAIS falsos positivos em média ({avg_chem_fp:.0f} vs {avg_dis_fp:.0f})\n\n")
        else:
            f.write(f"  ⚠️  Diseases geram MAIS falsos positivos em média ({avg_dis_fp:.0f} vs {avg_chem_fp:.0f})\n\n")
        
        f.write("Tabela de FP por Estratégia:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Estratégia':<15} {'Chem FP':<12} {'Dis FP':<12} {'Total FP':<12} {'Chem %':<10} {'Dis %':<10}\n")
        f.write("-"*100 + "\n")
        for _, row in df_results.iterrows():
            total_fp = row['chem_fp'] + row['dis_fp']
            chem_pct = (row['chem_fp'] / total_fp * 100) if total_fp > 0 else 0
            dis_pct = (row['dis_fp'] / total_fp * 100) if total_fp > 0 else 0
            f.write(f"{row['strategy']:<15} {int(row['chem_fp']):<12} {int(row['dis_fp']):<12} {int(total_fp):<12} {chem_pct:<10.2f} {dis_pct:<10.2f}\n")
        
        f.write("\nRanking de Falsos Positivos (FP) - Do maior para o menor:\n")
        f.write("-"*100 + "\n")
        df_results['total_fp'] = df_results['chem_fp'] + df_results['dis_fp']
        df_fp_sorted = df_results.sort_values(by='total_fp', ascending=False)
        for i, (_, row) in enumerate(df_fp_sorted.iterrows(), 1):
            total_fp = row['total_fp']
            f.write(f"  {i}. {row['strategy']:<15} : {int(total_fp)} FP (Chem: {int(row['chem_fp'])}, Dis: {int(row['dis_fp'])})\n")
        
        f.write("\nFALSOS NEGATIVOS (FN) - Análise por Tipo:\n")
        f.write("-"*100 + "\n")
        chem_fn_max = df_results.loc[df_results['chem_fn'].idxmax()]
        chem_fn_min = df_results.loc[df_results['chem_fn'].idxmin()]
        dis_fn_max = df_results.loc[df_results['dis_fn'].idxmax()]
        dis_fn_min = df_results.loc[df_results['dis_fn'].idxmin()]
        avg_chem_fn = df_results['chem_fn'].mean()
        avg_dis_fn = df_results['dis_fn'].mean()
        total_chem_fn = df_results['chem_fn'].sum()
        total_dis_fn = df_results['dis_fn'].sum()
        
        f.write(f"  Chemicals FN:\n")
        f.write(f"    Maior: {chem_fn_max['strategy']} com {int(chem_fn_max['chem_fn'])} FN ({chem_fn_max['chem_fn']/total_chem_fn*100:.2f}% do total)\n")
        f.write(f"    Menor: {chem_fn_min['strategy']} com {int(chem_fn_min['chem_fn'])} FN ({chem_fn_min['chem_fn']/total_chem_fn*100:.2f}% do total)\n\n")
        f.write(f"  Diseases FN:\n")
        f.write(f"    Maior: {dis_fn_max['strategy']} com {int(dis_fn_max['dis_fn'])} FN ({dis_fn_max['dis_fn']/total_dis_fn*100:.2f}% do total)\n")
        f.write(f"    Menor: {dis_fn_min['strategy']} com {int(dis_fn_min['dis_fn'])} FN ({dis_fn_min['dis_fn']/total_dis_fn*100:.2f}% do total)\n\n")
        
        if avg_chem_fn > avg_dis_fn:
            f.write(f"  ⚠️  Chemicals geram MAIS falsos negativos em média ({avg_chem_fn:.0f} vs {avg_dis_fn:.0f})\n\n")
        else:
            f.write(f"  ⚠️  Diseases geram MAIS falsos negativos em média ({avg_dis_fn:.0f} vs {avg_chem_fn:.0f})\n\n")
        
        f.write("Ranking de Falsos Negativos (FN) - Do maior para o menor:\n")
        f.write("-"*100 + "\n")
        df_results['total_fn'] = df_results['chem_fn'] + df_results['dis_fn']
        df_fn_sorted = df_results.sort_values(by='total_fn', ascending=False)
        for i, (_, row) in enumerate(df_fn_sorted.iterrows(), 1):
            total_fn = row['total_fn']
            f.write(f"  {i}. {row['strategy']:<15} : {int(total_fn)} FN (Chem: {int(row['chem_fn'])}, Dis: {int(row['dis_fn'])})\n")
        
        f.write("\nTRUE POSITIVES (TP) - Análise por Tipo:\n")
        f.write("-"*100 + "\n")
        chem_tp_max = df_results.loc[df_results['chem_tp'].idxmax()]
        chem_tp_min = df_results.loc[df_results['chem_tp'].idxmin()]
        dis_tp_max = df_results.loc[df_results['dis_tp'].idxmax()]
        dis_tp_min = df_results.loc[df_results['dis_tp'].idxmin()]
        
        f.write(f"  Chemicals TP:\n")
        f.write(f"    Maior: {chem_tp_max['strategy']} com {int(chem_tp_max['chem_tp'])} TP\n")
        f.write(f"    Menor: {chem_tp_min['strategy']} com {int(chem_tp_min['chem_tp'])} TP\n\n")
        f.write(f"  Diseases TP:\n")
        f.write(f"    Maior: {dis_tp_max['strategy']} com {int(dis_tp_max['dis_tp'])} TP\n")
        f.write(f"    Menor: {dis_tp_min['strategy']} com {int(dis_tp_min['dis_tp'])} TP\n\n")
        
        f.write("Ranking de True Positives (TP) - Do maior para o menor:\n")
        f.write("-"*100 + "\n")
        df_results['total_tp'] = df_results['chem_tp'] + df_results['dis_tp']
        df_tp_sorted = df_results.sort_values(by='total_tp', ascending=False)
        for i, (_, row) in enumerate(df_tp_sorted.iterrows(), 1):
            total_tp = row['total_tp']
            f.write(f"  {i}. {row['strategy']:<15} : {int(total_tp)} TP (Chem: {int(row['chem_tp'])}, Dis: {int(row['dis_tp'])})\n")
        
        f.write("\nResumo Comparativo - FP vs FN vs TP:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Estratégia':<15} {'TP':<12} {'FP':<12} {'FN':<12} {'TP/(TP+FP)':<15} {'TP/(TP+FN)':<15}\n")
        f.write("-"*100 + "\n")
        for _, row in df_results.iterrows():
            tp = row['overall_tp']
            fp = row['overall_fp']
            fn = row['overall_fn']
            precision_ratio = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_ratio = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f.write(f"{row['strategy']:<15} {int(tp):<12} {int(fp):<12} {int(fn):<12} {precision_ratio:<15.4f} {recall_ratio:<15.4f}\n")
        
        # ESTATÍSTICAS FINAIS
        f.write("\n" + "="*100 + "\n")
        f.write("ESTATÍSTICAS FINAIS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total de artigos analisados: {int(total_articles)}\n")
        f.write(f"Total de entidades Gold: {int(total_gold)}\n")
        f.write(f"Total de entidades Preditas: {int(total_pred)}\n")
    
    print(f"  ✅ Results.txt: {results_file}")

def find_models(base_dir: Path, prompt_type: Optional[str] = None) -> List[Dict]:
    """Encontra modelos"""
    indicios_dir = base_dir / "indicios_encontrados"
    models = []
    for model_dir in indicios_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for pt_dir in model_dir.iterdir():
            if not pt_dir.is_dir():
                continue
            if prompt_type and pt_dir.name != prompt_type:
                continue
            strategies = [s.name for s in pt_dir.iterdir() if s.is_dir()]
            if strategies:
                models.append({
                    'model_name': model_dir.name,
                    'prompt_type': pt_dir.name,
                    'prompt_type_dir': pt_dir,
                    'strategies': strategies
                })
    return models

def main():
    parser = argparse.ArgumentParser(description="Gera comparison, inferencias e results.txt")
    parser.add_argument('--model', type=str, default=None, help='Modelo específico')
    parser.add_argument('--prompt-type', type=str, default='type2', choices=['type1', 'type2'])
    parser.add_argument('--strategy', type=str, default=None, help='Estratégia específica')
    parser.add_argument('--base-dir', type=str, default=None)
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent
    gold_csv = base_dir / "data" / "cdr_gold.csv"
    
    if not gold_csv.exists():
        print(f"❌ Gold não encontrado: {gold_csv}")
        sys.exit(1)
    
    models = find_models(base_dir, args.prompt_type)
    if args.model:
        models = [m for m in models if m['model_name'] == args.model]
    
    print(f"📋 {len(models)} modelo(s)")
    
    for model_info in models:
        model_name = model_info['model_name']
        prompt_type = model_info['prompt_type']
        pt_dir = model_info['prompt_type_dir']
        strategies = [args.strategy] if args.strategy else [s for s in model_info['strategies'] if s in DEFAULT_STRATEGIES]
        
        print(f"\n{'='*80}")
        print(f"📊 {model_name} ({prompt_type})")
        print(f"{'='*80}")
        
        output_base = base_dir / "dataset" / model_name / prompt_type
        output_base.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        for strategy in strategies:
            strategy_dir = pt_dir / strategy
            if not strategy_dir.exists():
                continue
            result = process_strategy(strategy, strategy_dir, gold_csv, output_base, model_name, prompt_type)
            if result:
                all_results.append(result)
        
        if all_results:
            generate_results_txt(all_results, output_base, model_name, prompt_type)
            print(f"\n✅ Concluído: {model_name}")
    
    print(f"\n{'='*80}")
    print("✅ Processamento concluído!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
