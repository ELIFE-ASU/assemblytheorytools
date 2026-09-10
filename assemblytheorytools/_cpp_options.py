"""Validated search options for the parallelassemblycpp command line."""

from dataclasses import dataclass

_UINT64_MAX = (1 << 64) - 1
_INT_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class AssemblyCppOptions:
    """Configure the C++ calculator's search and diagnostic output.

    Parameters
    ----------
    runtime_ticks : int or None, default None
        Internal CPU-time budget, in ``std::clock`` ticks, from 0 through
        ``2**64 - 1``. ``None`` omits ``-runTime`` and uses the executable's
        unlimited default; ``2**64 - 1`` also means unlimited. This is distinct
        from the Python calculation's wall-clock ``timeout`` in seconds.
    enum_max : int or None, default None
        Maximum connected subgraphs in the initial graph enumeration, from 1
        through ``2**31 - 1``. Maps to ``-enumMax``; ``None`` uses the C++
        default (currently 50,000,000). Unavailable for string mode.
    pathway : bool, default True
        Write recovered pathway JSON. ``False`` passes ``--pathway=0``.
    accept_palindromes : bool, default False
        Treat a string fragment and its reversal as equivalent. Maps to
        ``-acceptPalindromes`` and is available only for string mode.
    parallel : {"off", "auto", "on"}, default "off"
        Select serial search, automatic parallel search with serial fallback,
        or require parallel search. Maps to ``--parallel``. ``"on"`` requires
        a compatible parallel build and cannot be combined with string mode,
        a finite ``runtime_ticks`` budget, or ``write_intermediate_mas=True``.
        ``"auto"`` allows C++ to fall back to serial for these cases.
    threads : int or {"auto"}, default "auto"
        OpenMP threads per process, from 1 through ``2**31 - 1``. Maps to
        ``--threads`` and applies when parallel search is enabled. Explicit
        thread counts are unavailable for string mode; availability depends
        on the executable and runtime environment.
    verbose : bool, default False
        Print the parsed graph into the calculation log (``--verbose=1``).
        Unavailable for string mode.
    memory_report : bool, default False
        Request the Linux peak-memory report (``-memTest=1``). C++ writes
        ``memUsage`` in the calculation directory.
    telemetry : bool, default False
        Write ``INPUTTelemetry.json`` (``--telemetry=1``). Requires an
        executable built with telemetry support and is unavailable for
        string mode. The flag is omitted entirely when disabled, so ordinary
        builds work without telemetry support.
    write_intermediate_mas : bool, default False
        Write index improvements to ``INPUTIntermediateMAs`` using
        ``-writeIntermediateMAs=1``. Unavailable for string mode and forced
        parallel search.

    Notes
    -----
    Options are immutable and strictly typed; booleans are not accepted as
    integer values. Incompatible input modes are checked before execution.
    Default controls are omitted where possible to support older calculators;
    explicitly selecting a newer feature requires a supporting executable.
    Older camelCase aliases are used for renamed C++ flags.

    Hydrogen removal and disconnected-component correction are managed by
    Python, so the bridge always passes ``-removeHydrogens=0`` and
    ``-compensateDisjoint=0``. The input API selects ``-runStrings=1`` for
    string mode. Diagnostic file options retain the calculation directory;
    use the calculation's ``return_log_file=True`` to obtain its location.
    """

    runtime_ticks: int | None = None
    enum_max: int | None = None
    pathway: bool = True
    accept_palindromes: bool = False
    parallel: str = "off"
    threads: int | str = "auto"
    verbose: bool = False
    memory_report: bool = False
    telemetry: bool = False
    write_intermediate_mas: bool = False

    def __post_init__(self) -> None:
        for name in (
            "pathway", "accept_palindromes", "verbose", "memory_report",
            "telemetry", "write_intermediate_mas",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

        for name, minimum, maximum in (
            ("runtime_ticks", 0, _UINT64_MAX),
            ("enum_max", 1, _INT_MAX),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not int:
                raise TypeError(f"{name} must be an int or None")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be from {minimum} to {maximum}")

        if type(self.parallel) is not str:
            raise TypeError("parallel must be 'auto', 'on', or 'off'")
        if self.parallel not in {"auto", "on", "off"}:
            raise ValueError("parallel must be 'auto', 'on', or 'off'")

        if type(self.threads) is str:
            if self.threads != "auto":
                raise ValueError("threads must be 'auto' or a positive int")
        elif type(self.threads) is not int:
            raise TypeError("threads must be 'auto' or a positive int")
        elif not 1 <= self.threads <= _INT_MAX:
            raise ValueError(f"threads must be from 1 to {_INT_MAX}")

    @property
    def _retain_files(self) -> bool:
        return self.memory_report or self.telemetry or self.write_intermediate_mas

    def _validate_mode(self, *, str_mode: bool) -> None:
        """Reject controls that the selected input mode cannot honor."""
        if str_mode:
            if self.parallel == "on":
                raise ValueError("parallel='on' is unavailable for C++ string mode")
            for name, configured in (
                ("enum_max", self.enum_max is not None),
                ("threads", self.threads != "auto"),
                ("verbose", self.verbose),
                ("telemetry", self.telemetry),
                ("write_intermediate_mas", self.write_intermediate_mas),
            ):
                if configured:
                    raise ValueError(f"{name} is unavailable for C++ string mode")
        else:
            if self.accept_palindromes:
                raise ValueError("accept_palindromes is available only for C++ string mode")
            if self.parallel == "on":
                if self.runtime_ticks not in (None, _UINT64_MAX):
                    raise ValueError("parallel='on' cannot use a finite runtime_ticks budget")
                if self.write_intermediate_mas:
                    raise ValueError("parallel='on' cannot use write_intermediate_mas")

    def _arguments(self, *, str_mode: bool) -> list[str]:
        """Validate the input mode and serialize each applicable option once."""
        self._validate_mode(str_mode=str_mode)
        arguments = [
            "-removeHydrogens=0",
            "-compensateDisjoint=0",
            f"-memTest={int(self.memory_report)}",
        ]
        if str_mode:
            arguments.append("-runStrings=1")
        if self.runtime_ticks is not None:
            arguments.append(f"-runTime={self.runtime_ticks}")
        if self.enum_max is not None:
            arguments.append(f"-enumMax={self.enum_max}")
        if not self.pathway:
            arguments.append("--pathway=0")
        if self.accept_palindromes:
            arguments.append("-acceptPalindromes=1")
        if self.parallel != "off":
            arguments.append(f"--parallel={self.parallel}")
        if self.threads != "auto":
            arguments.append(f"--threads={self.threads}")
        if self.verbose:
            arguments.append("--verbose=1")
        if self.telemetry:
            arguments.append("--telemetry=1")
        if self.write_intermediate_mas:
            arguments.append("-writeIntermediateMAs=1")
        return arguments
