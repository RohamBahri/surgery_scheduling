# =============================================================================
# DATAFRAME COLUMN NAMES
# =============================================================================


class DataColumns:
    """Primary DataFrame column names (standardized snake_case)."""

    # Identifiers
    PATIENT_ID = "patient_id"
    PATIENT_TYPE = "patient_type"
    CASE_SERVICE = "case_service"
    MAIN_PROCEDURE = "main_procedure"
    MAIN_PROCEDURE_ID = "main_procedure_id"
    OPERATING_ROOM = "operating_room"
    SURGEON = "surgeon"
    SURGEON_CODE = "surgeon_code"

    # Timestamps
    CONSULT_DATE = "consult_date"
    DECISION_DATE = "decision_date"
    ACTUAL_START = "actual_start"
    ACTUAL_STOP = "actual_stop"
    ENTER_ROOM = "enter_room"
    LEAVE_ROOM = "leave_room"

    # Date/time components (for parsing)
    ACTUAL_START_DATE = "actual_start_date"
    ACTUAL_START_TIME = "actual_start_time"
    ACTUAL_STOP_DATE = "actual_stop_date"
    ACTUAL_STOP_TIME = "actual_stop_time"
    ENTER_ROOM_DATE = "enter_room_date"
    ENTER_ROOM_TIME = "enter_room_time"
    LEAVE_ROOM_DATE = "leave_room_date"
    LEAVE_ROOM_TIME = "leave_room_time"

    # Durations
    BOOKED_MIN = "booked_time_minutes"
    PROCEDURE_DURATION_MIN = "procedure_duration_min"
    PREPARATION_DURATION_MIN = "preparation_duration_min"

    # Time features (derived)
    WEEK_OF_YEAR = "week_of_year"
    MONTH = "month"
    YEAR = "year"


class ScheduleColumns:
    """Column names for schedule output DataFrames."""

    SURGERY_ID = "SurgeryID"
    PROC_ID = "ProcID"
    ASSIGNED_DAY = "AssignedDay"
    ASSIGNED_BLOCK = "AssignedBlock"
    ACTUAL_DUR_MIN = "ActualDurMin"


# =============================================================================
# MACHINE LEARNING FEATURES
# =============================================================================


class FeatureColumns:
    """Feature sets for machine learning models."""

    NUMERIC = [
        DataColumns.BOOKED_MIN,
        DataColumns.WEEK_OF_YEAR,
        DataColumns.MONTH,
        DataColumns.YEAR,
    ]

    CATEGORICAL = [
        DataColumns.PATIENT_TYPE,
        DataColumns.MAIN_PROCEDURE_ID,
        DataColumns.SURGEON_CODE,
        DataColumns.CASE_SERVICE,
    ]

    ALL = NUMERIC + CATEGORICAL

    # For theta predictor (raw feature names)
    THETA_NUMERIC_RAW = [
        DataColumns.BOOKED_MIN,
        DataColumns.WEEK_OF_YEAR,
        DataColumns.MONTH,
        DataColumns.YEAR,
    ]


# =============================================================================
# DOMAIN CONSTANTS
# =============================================================================


class DomainConstants:
    """Domain-specific numerical and categorical constants."""

    # Duration limits
    MIN_PROCEDURE_DURATION = 30
    MAX_OVERTIME_MINUTES_PER_BLOCK = 240
    DEFAULT_BLOCK_SIZE_MINUTES = 480

    # Time constants
    SECONDS_PER_MINUTE = 60
    DEFAULT_TIME_STR = "00:00:00"

    # Categories
    OTHER_CATEGORY = "Other"
    UNKNOWN_CATEGORY = "Unknown"
    EMERGENCY_PATIENT_CATEGORY = "EMERGENCY PATIENT"
    OR_ROOM_PREFIX = "OR"

    # SAA
    DEFAULT_SAA_RANDOM_SEED = 42


# =============================================================================
# GUROBI CONSTANTS
# =============================================================================


class GurobiConstants:
    """Gurobi-related constants and variable prefixes."""

    # Variable prefixes
    VAR_X_PREFIX = "x"
    VAR_R_PREFIX = "r"
    VAR_OT_PREFIX = "ot"
    VAR_IT_PREFIX = "it"
    VAR_ZSPILL_PREFIX = "zspill"
    VAR_THETA_PREFIX = "theta"
    VAR_ETA_PREFIX = "eta"
    VAR_ABS_THETA_PREFIX = "abs_theta"
    VAR_BLK_SIGNED_ERR_PREFIX = "blk_signed_err"
    VAR_BLK_ABS_ERR_PREFIX = "blk_abs_err"
    VAR_U_LEARN_ABS_ERROR_PREFIX = "u_abs_learn_err"

    # Parameter values
    OUTPUT_FLAG_SILENT = 0
    OUTPUT_FLAG_VERBOSE = 1
    THREADS_ALL_CORES = 0
    PRESOLVE_AUTO = -1
    MIP_FOCUS_BALANCED = 0
    MIP_FOCUS_FEASIBILITY = 1
    MIP_FOCUS_OPTIMALITY = 2
    MIP_FOCUS_BOUND = 3
    LAZY_CONSTRAINTS_ENABLED = 1

    # Other
    DEFAULT_MODEL_NAME = "GurobiModel"
    BINARY_THRESHOLD = 0.5
    REGEX_NUMERICAL_VALUES = r"\d+"


# =============================================================================
# INTERNAL KEYS
# =============================================================================


class InternalKeys:
    """Internal cache keys and parameter dictionary keys."""

    BFV_MU_SIGMA_CACHE_KEY = "bfv_mu_sigma_scaling_cache"
    HORIZON_START_DATE_PARAM_KEY = "_internal_horizon_start_date"


# =============================================================================
# JSON OUTPUT STRUCTURE
# =============================================================================


class JSONKeys:
    """Keys for structuring JSON output files."""

    # Main structure
    CONFIG = "config"
    HORIZONS = "horizons"
    AGGREGATE = "aggregate"

    # Config sub-keys
    CONFIG_SAA_SCENARIOS = "saa_scenarios_configured"
    CONFIG_NUM_HORIZONS = "num_horizons_planned"

    # Per-horizon keys
    HORIZON_INDEX = "horizon_index"
    START_DATE = "start_date"
    STATUS = "solver_status"
    PLANNED_COST = "planned_objective_cost"
    ACTUAL_COST = "actual_total_cost"
    OVERTIME_MIN = "overtime_minutes_total"
    IDLE_MIN = "idle_minutes_total"
    SCHEDULED_COUNT = "scheduled_surgeries_count"
    REJECTED_COUNT = "rejected_surgeries_count"
    RUNTIME_SEC = "solver_runtime_seconds"

    # Aggregate keys
    MEDIAN_PLANNED_COST = "median_planned_objective_cost"
    AVG_PLANNED_COST = "average_planned_objective_cost"
    MEDIAN_ACTUAL_COST = "median_actual_total_cost"
    AVG_ACTUAL_COST = "average_actual_total_cost"
    MEDIAN_OVERTIME_MIN = "median_overtime_minutes"
    AVG_OVERTIME_MIN = "average_overtime_minutes"
    MEDIAN_IDLE_MIN = "median_idle_minutes"
    AVG_IDLE_MIN = "average_idle_minutes"
    AVG_RUNTIME_SEC = "average_runtime_seconds"
    MEDIAN_SCHEDULED = "median_scheduled_surgeries"
    AVG_SCHEDULED = "average_scheduled_surgeries"
    MEDIAN_REJECTED = "median_rejected_surgeries"
    AVG_REJECTED = "average_rejected_surgeries"
    TOTAL_BLOCKS = "total_blocks"


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


class LoggingConstants:
    """Logging configuration constants."""

    DEFAULT_LOGGER_NAME = "SurgerySchedulingApp"
    LOG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
    )
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
