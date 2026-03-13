# tools/builtin/system.py
"""
Outils système builtin — READ_ONLY, sans effet de bord.

Tous les outils sont async. Les opérations bloquantes (psutil.cpu_percent,
socket I/O) sont déléguées à asyncio.to_thread() pour ne pas bloquer
la boucle d'événements FastAPI.

Sécurité :
    - hostname, IP, username : filtrés (principe du moindre privilège)
    - subprocess : interdit (socket direct pour ping)
    - Aucune écriture fichier ni mutation système

Format réponse :
    get_current_time → str  (ISO 8601 direct ou "Error: ..." sur échec)
    get_system_info  → dict (always, avec champ "error" si échec)
    ping_host        → dict (always, avec champ "success" bool)
"""

from __future__ import annotations

import asyncio
import socket
import time
from datetime import datetime
from typing import Any

import psutil
import pytz

from core.tools import PermissionLevel, ToolParameter, ToolSchema


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


async def get_current_time(timezone: str = "UTC") -> str:
    """
    Retourne l'heure actuelle dans le fuseau horaire spécifié au format ISO 8601.

    Args:
        timezone: Identifiant IANA du fuseau horaire (ex: 'Europe/Paris',
                  'America/New_York', 'Asia/Tokyo', 'UTC').
                  Liste complète : https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Returns:
        str: Heure au format ISO 8601 avec offset (ex: '2026-03-13T14:30:00+01:00').
             En cas de timezone invalide : "Error: Unknown timezone 'XYZ'"

    Exemples LLM :
        get_current_time("Europe/Paris")   → "2026-03-13T14:30:00+01:00"
        get_current_time("UTC")            → "2026-03-13T13:30:00+00:00"
        get_current_time("America/Tokyo")  → "Error: Unknown timezone 'America/Tokyo'"
    """
    try:
        tz = pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Use IANA format (e.g. 'Europe/Paris', 'UTC')."

    now = datetime.now(tz)
    return now.isoformat()


# ---------------------------------------------------------------------------
# get_system_info
# ---------------------------------------------------------------------------


def _collect_system_info() -> dict[str, Any]:
    """Collecte synchrone des métriques système (exécutée dans un thread)."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu = psutil.cpu_percent(interval=0.1)   # bloque 100ms — dans thread

    return {
        "cpu_percent": cpu,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "memory_percent": mem.percent,
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_percent": disk.percent,
        "platform": psutil.os.uname().sysname if hasattr(psutil.os, "uname") else "unknown",
        "cpu_count": psutil.cpu_count(logical=True),
        # hostname, IP, username délibérément exclus (principe du moindre privilège)
    }


async def get_system_info() -> dict[str, Any]:
    """
    Retourne les métriques système actuelles (CPU, mémoire, disque).

    Returns:
        dict avec les champs :
            cpu_percent      (float) : Utilisation CPU en % sur 0.1s
            memory_used_gb   (float) : RAM utilisée en Go
            memory_total_gb  (float) : RAM totale en Go
            memory_percent   (float) : % RAM utilisée
            disk_used_gb     (float) : Disque utilisé en Go (partition /)
            disk_total_gb    (float) : Disque total en Go
            disk_percent     (float) : % disque utilisé
            platform         (str)   : Système d'exploitation (ex: 'Darwin')
            cpu_count        (int)   : Nombre de cœurs logiques
        En cas d'erreur : {"error": "message d'erreur"}

    Note sécurité :
        hostname, adresses IP et nom d'utilisateur ne sont pas retournés.

    Exemple LLM :
        get_system_info() → {
            "cpu_percent": 23.5, "memory_used_gb": 12.4,
            "memory_total_gb": 32.0, "memory_percent": 38.8, ...
        }
    """
    try:
        return await asyncio.to_thread(_collect_system_info)
    except Exception as exc:
        return {"error": f"Erreur collecte métriques : {exc}"}


# ---------------------------------------------------------------------------
# ping_host
# ---------------------------------------------------------------------------


def _tcp_ping(host: str, timeout: float) -> dict[str, Any]:
    """
    Teste la connectivité TCP vers un hôte. Tente les ports 80, 443 puis 53.
    Opération bloquante — exécutée dans un thread via asyncio.to_thread().
    """
    start = time.perf_counter()

    # Résolution DNS d'abord (valide même si tous les ports sont filtrés)
    try:
        resolved = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return {
            "success": False,
            "host": host,
            "error": f"Hôte introuvable (DNS) : {exc}",
        }

    # Tentative TCP sur ports communs dans l'ordre
    last_error: str = ""
    for port in (80, 443, 53):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "success": True,
                    "host": host,
                    "port": port,
                    "latency_ms": latency_ms,
                }
        except socket.timeout:
            last_error = f"Timeout ({timeout}s) sur port {port}"
        except ConnectionRefusedError:
            # Port fermé mais hôte joignable — succès partiel
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "success": True,
                "host": host,
                "port": port,
                "latency_ms": latency_ms,
                "note": f"Port {port} fermé (hôte joignable)",
            }
        except OSError as exc:
            last_error = str(exc)

    # DNS résolu mais tous les ports TCP échouent — hôte probablement ICMP-only
    # (ex: routeur qui répond au ping mais bloque TCP)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "success": True,
        "host": host,
        "latency_ms": latency_ms,
        "note": "DNS résolu, ports TCP filtrés. Hôte probablement joignable.",
        "resolved_addresses": [str(r[4][0]) for r in resolved[:3]],
    }


async def ping_host(host: str, timeout: float = 5.0) -> dict[str, Any]:
    """
    Teste la connectivité réseau vers un hôte (IP ou nom de domaine).

    Méthode : TCP connect sur ports 80, 443, 53 (dans l'ordre). Si tous
    échouent mais le DNS résout, l'hôte est considéré joignable (ports filtrés).
    N'utilise pas ICMP (nécessite root) ni subprocess.

    Args:
        host:    Adresse IP (ex: '8.8.8.8') ou nom de domaine (ex: 'google.com').
        timeout: Délai maximum par tentative en secondes (défaut: 5.0).

    Returns:
        dict avec les champs :
            success      (bool)  : True si l'hôte est joignable
            host         (str)   : Hôte testé
            latency_ms   (float) : Latence mesurée en ms (si success=True)
            port         (int)   : Port TCP utilisé (si connexion établie)
            error        (str)   : Message d'erreur (si success=False)
            note         (str)   : Information complémentaire (optionnel)

    Exemples LLM :
        ping_host("8.8.8.8")           → {"success": True, "latency_ms": 12.3, ...}
        ping_host("definitely.invalid") → {"success": False, "error": "Hôte introuvable..."}
    """
    try:
        return await asyncio.to_thread(_tcp_ping, host, timeout)
    except Exception as exc:
        return {
            "success": False,
            "host": host,
            "error": f"Erreur inattendue : {exc}",
        }


# ---------------------------------------------------------------------------
# Catalogue builtin — source unique de vérité pour schemas + executors
# ---------------------------------------------------------------------------
#
# Format : name → (ToolSchema, async Callable)
# register_builtin_tools() itère sur ce dict.
#

BUILTIN_TOOLS: dict[str, tuple[ToolSchema, Any]] = {
    "get_current_time": (
        ToolSchema(
            name="get_current_time",
            description=(
                "Get the current date and time in a specific timezone. "
                "Returns an ISO 8601 formatted string with UTC offset "
                "(e.g. '2026-03-13T14:30:00+01:00'). "
                "Use IANA timezone names like 'Europe/Paris', 'America/New_York', 'UTC'."
            ),
            parameters=[
                ToolParameter(
                    name="timezone",
                    type="string",
                    description=(
                        "IANA timezone identifier. Examples: 'Europe/Paris', "
                        "'America/New_York', 'Asia/Tokyo', 'UTC'. "
                        "Defaults to 'UTC' if omitted."
                    ),
                    required=False,
                )
            ],
            permission_level=PermissionLevel.READ_ONLY,
            requires_confirmation=False,
            category="system",
            created_by="system",
            version="1.0.0",
        ),
        get_current_time,
    ),
    "get_system_info": (
        ToolSchema(
            name="get_system_info",
            description=(
                "Get current system resource usage: CPU percentage, RAM usage (used/total in GB), "
                "disk usage (used/total in GB), OS platform, and CPU core count. "
                "Useful for monitoring or answering questions about system performance."
            ),
            parameters=[],
            permission_level=PermissionLevel.READ_ONLY,
            requires_confirmation=False,
            category="system",
            created_by="system",
            version="1.0.0",
        ),
        get_system_info,
    ),
    "ping_host": (
        ToolSchema(
            name="ping_host",
            description=(
                "Test network connectivity to a host (IP address or domain name) using TCP. "
                "Returns success status and latency in milliseconds. "
                "Uses TCP ports 80/443/53 — does not require root privileges."
            ),
            parameters=[
                ToolParameter(
                    name="host",
                    type="string",
                    description=(
                        "IP address (e.g. '8.8.8.8') or domain name (e.g. 'google.com') to test."
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Maximum wait time per attempt in seconds. Default: 5.0.",
                    required=False,
                ),
            ],
            permission_level=PermissionLevel.READ_ONLY,
            requires_confirmation=False,
            category="system",
            created_by="system",
            version="1.0.0",
        ),
        ping_host,
    ),
}
