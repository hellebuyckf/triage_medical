"""Benchmark de latence pour l'API de triage médical en conditions réalistes.

Ce script mesure les performances de l'endpoint POST /triage en exécutant
des requêtes séquentielles (baseline) puis concurrentes (charge) contre
une API en cours d'exécution.

Métriques rapportées : min, max, mean, P50, P95, P99, taux d'erreur.
Un seuil SLA sur le P95 séquentiel peut être asserté (exit code 1 si dépassé).

Usage:
    uv run python scripts/serving/benchmark.py [options]

Options:
    --url           URL de base de l'API (défaut: http://localhost:8080)
    --n-requests    Nombre de requêtes séquentielles (défaut: 20)
    --concurrency   Niveau de concurrence (défaut: 5)
    --p95-max-ms    Seuil SLA P95 en millisecondes (défaut: 5000)
    --no-report     Désactiver la sauvegarde du rapport Markdown
    --verbose       Mode verbeux (affiche chaque requête)

Example:
    # Benchmark par défaut contre l'API locale
    uv run python scripts/serving/benchmark.py

    # Vérifier que P95 ≤ 3 s avec 30 requêtes, 8 en parallèle
    uv run python scripts/serving/benchmark.py --n-requests 30 --concurrency 8 --p95-max-ms 3000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import httpx

# ── Cas de test ───────────────────────────────────────────────────────────────

# 10 scénarios représentatifs couvrant les 3 niveaux d'urgence.
_SYMPTOM_PROMPTS: list[str] = [
    # max (4 cas)
    "Douleur thoracique intense avec sudation profuse et douleur irradiant dans le bras gauche depuis 20 minutes.",
    "AVC suspecté : paralysie faciale droite, difficultés à parler, apparition brutale il y a 10 minutes.",
    "Difficultés respiratoires sévères, SpO2 à 88 %, historique d'asthme, inhalateur inefficace.",
    "Perte de connaissance lors d'un effort physique, brève mais totale, sans traumatisme.",
    # moderate (3 cas)
    "Douleur abdominale modérée depuis hier soir, légère fièvre à 38 °C, selles normales.",
    "Maux de tête intenses depuis 6 heures, pas de fièvre, pas de vomissements, antécédent de migraine.",
    "Plaie lacérée au bras gauche (environ 4 cm), saignement contrôlé, vaccin tétanos à jour.",
    # deferred (3 cas)
    "Fièvre à 38,2 °C depuis deux jours, toux sèche légère, pas de difficultés respiratoires.",
    "Entorse de cheville droite suite à une chute, douleur à la palpation, légère tuméfaction.",
    "Renouvellement d'ordonnance d'antihypertenseur, tension bien contrôlée, pas de symptôme aigu.",
]


# ── Types ─────────────────────────────────────────────────────────────────────


class RequestResult(NamedTuple):
    """Résultat d'une requête individuelle vers /triage.

    Attributes:
        latency_ms: Temps de réponse mesuré côté client (ms).
        status_code: Code HTTP retourné (0 si erreur réseau).
        urgency_level: Niveau d'urgence parsé par l'API (None si erreur).
        error: Message d'erreur si la requête a échoué, sinon None.
    """

    latency_ms: float
    status_code: int
    urgency_level: str | None
    error: str | None


# ── Statistiques ──────────────────────────────────────────────────────────────


def _percentile(sorted_data: list[float], p: float) -> float:
    """Retourne le p-ième percentile d'une liste déjà triée.

    Args:
        sorted_data: Liste de valeurs triées par ordre croissant.
        p: Percentile voulu (0–100).

    Returns:
        Valeur au percentile demandé.
    """
    if not sorted_data:
        return 0.0
    idx = min(int(p / 100 * len(sorted_data)), len(sorted_data) - 1)
    return sorted_data[idx]


def compute_stats(latencies: list[float]) -> dict[str, float]:
    """Calcule les statistiques descriptives sur une liste de latences.

    Args:
        latencies: Liste des latences en millisecondes (requêtes réussies uniquement).

    Returns:
        Dictionnaire avec count, min, max, mean, median, p95, p99 (en ms).
    """
    if not latencies:
        return {"count": 0}
    s = sorted(latencies)
    return {
        "count": float(len(s)),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
        "mean_ms": round(statistics.mean(s), 1),
        "median_ms": round(statistics.median(s), 1),
        "p95_ms": round(_percentile(s, 95), 1),
        "p99_ms": round(_percentile(s, 99), 1),
    }


# ── Client ────────────────────────────────────────────────────────────────────


async def _send_request(
    client: httpx.AsyncClient,
    symptoms: str,
    timeout: float = 60.0,
) -> RequestResult:
    """Envoie une requête POST /triage et mesure la latence côté client.

    Args:
        client: Client HTTP asynchrone.
        symptoms: Description des symptômes à envoyer.
        timeout: Timeout en secondes avant abandon.

    Returns:
        RequestResult avec latence, code HTTP, urgency_level et éventuelle erreur.
    """
    t0 = time.monotonic()
    try:
        response = await client.post(
            "/triage",
            json={"symptoms": symptoms},
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if response.status_code == 200:
            data = response.json()
            return RequestResult(
                latency_ms=latency_ms,
                status_code=200,
                urgency_level=data.get("urgency_level"),
                error=None,
            )
        return RequestResult(
            latency_ms=latency_ms,
            status_code=response.status_code,
            urgency_level=None,
            error=f"HTTP {response.status_code}: {response.text[:120]}",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return RequestResult(
            latency_ms=latency_ms,
            status_code=0,
            urgency_level=None,
            error=str(exc)[:120],
        )


# ── Phases ────────────────────────────────────────────────────────────────────


async def run_sequential(
    client: httpx.AsyncClient,
    n_requests: int,
    logger: logging.Logger,
) -> list[RequestResult]:
    """Phase 1 — requêtes séquentielles (une à la fois).

    Mesure la latence de base sans contention.

    Args:
        client: Client HTTP asynchrone.
        n_requests: Nombre total de requêtes à envoyer.
        logger: Logger.

    Returns:
        Liste ordonnée des résultats.
    """
    results: list[RequestResult] = []
    # Cycle sur les prompts si n_requests > len(_SYMPTOM_PROMPTS)
    prompts = [_SYMPTOM_PROMPTS[i % len(_SYMPTOM_PROMPTS)] for i in range(n_requests)]

    logger.info("Phase séquentielle : %d requêtes...", n_requests)
    for i, symptoms in enumerate(prompts, start=1):
        result = await _send_request(client, symptoms)
        results.append(result)
        if result.error is None:
            logger.debug(
                "  [%02d/%02d] ✓ %.0f ms  [%s]",
                i,
                n_requests,
                result.latency_ms,
                result.urgency_level,
            )
        else:
            logger.debug("  [%02d/%02d] ✗ %s", i, n_requests, result.error)

    return results


async def run_concurrent(
    client: httpx.AsyncClient,
    concurrency: int,
    logger: logging.Logger,
) -> list[RequestResult]:
    """Phase 2 — requêtes simultanées (charge).

    Simule plusieurs utilisateurs envoyant une requête en même temps.

    Args:
        client: Client HTTP asynchrone.
        concurrency: Nombre de requêtes lancées simultanément.
        logger: Logger.

    Returns:
        Liste des résultats (ordre non garanti).
    """
    logger.info("Phase concurrente : %d requêtes simultanées...", concurrency)
    tasks = [
        _send_request(client, _SYMPTOM_PROMPTS[i % len(_SYMPTOM_PROMPTS)])
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── Rapport Markdown ──────────────────────────────────────────────────────────


def _render_stats_table(stats: dict[str, float], n_total: int, n_errors: int) -> list[str]:
    """Génère les lignes d'un tableau Markdown de statistiques.

    Args:
        stats: Dictionnaire retourné par compute_stats().
        n_total: Nombre total de requêtes (succès + erreurs).
        n_errors: Nombre de requêtes en erreur.

    Returns:
        Liste de lignes Markdown.
    """
    labels = {
        "count": "n (succès)",
        "min_ms": "min (ms)",
        "max_ms": "max (ms)",
        "mean_ms": "mean (ms)",
        "median_ms": "P50 (ms)",
        "p95_ms": "P95 (ms)",
        "p99_ms": "P99 (ms)",
    }
    lines = ["| Métrique | Valeur |", "|---|---|"]
    for key, label in labels.items():
        if key in stats:
            lines.append(f"| {label} | {stats[key]} |")
    lines.append(f"| erreurs | {n_errors} / {n_total} |")
    return lines


def format_report(
    *,
    url: str,
    seq_results: list[RequestResult],
    conc_results: list[RequestResult],
    p95_max_ms: float,
    timestamp: str,
) -> str:
    """Génère le rapport Markdown complet du benchmark.

    Args:
        url: URL de l'API testée.
        seq_results: Résultats de la phase séquentielle.
        conc_results: Résultats de la phase concurrente.
        p95_max_ms: Seuil SLA P95 (ms) à vérifier.
        timestamp: Horodatage ISO 8601 UTC.

    Returns:
        Rapport formaté en Markdown (prêt à écrire dans un fichier).
    """
    seq_ok = [r for r in seq_results if r.error is None]
    conc_ok = [r for r in conc_results if r.error is None]

    seq_stats = compute_stats([r.latency_ms for r in seq_ok])
    conc_stats = compute_stats([r.latency_ms for r in conc_ok])

    sla_pass = seq_stats.get("p95_ms", float("inf")) <= p95_max_ms
    sla_badge = "✅ PASS" if sla_pass else "❌ FAIL"

    lines: list[str] = [
        "# Benchmark de latence — API Triage CHSA",
        "",
        f"**Date** : {timestamp}  ",
        f"**URL** : `{url}`  ",
        f"**SLA P95 séquentiel** : ≤ {p95_max_ms:.0f} ms → {sla_badge}",
        "",
        "---",
        "",
        f"## Phase séquentielle ({len(seq_results)} requêtes, une à la fois)",
        "",
        *_render_stats_table(seq_stats, len(seq_results), len(seq_results) - len(seq_ok)),
        "",
        f"## Phase concurrente ({len(conc_results)} requêtes simultanées)",
        "",
        *_render_stats_table(conc_stats, len(conc_results), len(conc_results) - len(conc_ok)),
        "",
        "---",
        "",
        "## Distribution des niveaux d'urgence (phase séquentielle)",
        "",
    ]

    counts = Counter(r.urgency_level for r in seq_ok)
    for level in ["max", "moderate", "deferred", None]:
        if level in counts:
            label = level if level is not None else "None (non parsé)"
            lines.append(f"- `{label}` : {counts[level]} / {len(seq_ok)}")

    lines += [
        "",
        "---",
        "",
        "*Généré par `scripts/serving/benchmark.py`*",
    ]
    return "\n".join(lines)


# ── Entrée principale ─────────────────────────────────────────────────────────


async def main() -> int:
    """Orchestre le benchmark et retourne un code de sortie.

    Returns:
        0 si le SLA P95 est respecté (ou non asserté), 1 sinon.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark de latence de l'API Triage CHSA en conditions réalistes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8080", help="URL de base de l'API")
    parser.add_argument(
        "--n-requests", type=int, default=20, help="Nombre de requêtes séquentielles"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Requêtes simultanées (phase concurrente)"
    )
    parser.add_argument(
        "--p95-max-ms", type=float, default=5000.0, help="Seuil SLA P95 en ms (exit 1 si dépassé)"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Ne pas sauvegarder le rapport Markdown"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Afficher le détail de chaque requête"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    logger = logging.getLogger("benchmark")

    # ── Health check ──────────────────────────────────────────────────────────
    logger.info("Vérification de l'API à %s ...", args.url)
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=10.0) as probe:
            resp = await probe.get("/health")
        health = resp.json()
        if resp.status_code != 200 or health.get("status") != "ok":
            logger.error(
                "API non disponible (status=%s, body=%s). Démarrez-la d'abord.",
                resp.status_code,
                health,
            )
            return 1
        logger.info("API opérationnelle — modèle : %s", health.get("model", "inconnu"))
    except Exception as exc:
        logger.error("Impossible de contacter l'API : %s", exc)
        logger.error("Démarrez l'API avec : make serve-local")
        return 1

    # ── Benchmark ─────────────────────────────────────────────────────────────
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient(base_url=args.url, timeout=120.0) as client:
        seq_results = await run_sequential(client, args.n_requests, logger)
        conc_results = await run_concurrent(client, args.concurrency, logger)

    # ── Résumé console ────────────────────────────────────────────────────────
    seq_ok = [r for r in seq_results if r.error is None]
    conc_ok = [r for r in conc_results if r.error is None]
    seq_stats = compute_stats([r.latency_ms for r in seq_ok])
    conc_stats = compute_stats([r.latency_ms for r in conc_ok])

    logger.info("─" * 60)
    logger.info(
        "SÉQUENTIEL  P50=%.0f ms  P95=%.0f ms  P99=%.0f ms  (n=%d, err=%d/%d)",
        seq_stats.get("median_ms", 0),
        seq_stats.get("p95_ms", 0),
        seq_stats.get("p99_ms", 0),
        len(seq_ok),
        len(seq_results) - len(seq_ok),
        len(seq_results),
    )
    logger.info(
        "CONCURRENT  P50=%.0f ms  P95=%.0f ms               (n=%d, err=%d/%d)",
        conc_stats.get("median_ms", 0),
        conc_stats.get("p95_ms", 0),
        len(conc_ok),
        len(conc_results) - len(conc_ok),
        len(conc_results),
    )

    sla_pass = seq_stats.get("p95_ms", float("inf")) <= args.p95_max_ms
    if sla_pass:
        logger.info("SLA P95 ≤ %.0f ms → ✅ PASS", args.p95_max_ms)
    else:
        logger.error(
            "SLA P95 ≤ %.0f ms → ❌ FAIL (mesuré : %.0f ms)",
            args.p95_max_ms,
            seq_stats.get("p95_ms", 0),
        )

    # ── Rapport ───────────────────────────────────────────────────────────────
    if not args.no_report:
        report_md = format_report(
            url=args.url,
            seq_results=seq_results,
            conc_results=conc_results,
            p95_max_ms=args.p95_max_ms,
            timestamp=timestamp,
        )
        report_dir = Path("reports/serving")
        report_dir.mkdir(parents=True, exist_ok=True)
        slug = timestamp.replace(":", "-").replace("T", "_").replace("Z", "")
        report_path = report_dir / f"benchmark_{slug}.md"
        report_path.write_text(report_md, encoding="utf-8")
        logger.info("Rapport sauvegardé → %s", report_path)

    return 0 if sla_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
