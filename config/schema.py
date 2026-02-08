
import yaml
from pydantic import BaseModel, ConfigDict, Field

# --- Data Models ---

class Point(BaseModel):
    """Point coordinates."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)

    # Pydantic V2 style configuration
    model_config = ConfigDict(
        json_schema_extra={
            "options": {
                "disable_collapse": True,
                "disable_edit_json": True,
                "disable_properties": True
            }
        }
    )


class Zone(BaseModel):
    """Defines a polygon zone of interest within a video stream."""
    name: str
    # List of points - use Point objects for cleaner JSON schema
    points: list[Point] = Field(
        default_factory=list,
        json_schema_extra={
            "options": {
                "collapsed": True,
                "disable_array_reorder": False
            }
        }
    )

    # Optional: Specific actions to monitor in this zone
    triggers: list[str] = Field(default_factory=lambda: ["person"])


class Door(BaseModel):
    """
    Defines a door/threshold for detecting enter/exit direction.

    A door is defined by 4 points forming a polygon (the door area) and a
    user-defined normal angle that indicates the "enter" direction.

    Movement in the direction of the normal is considered "entering",
    movement opposite (180°) is considered "exiting".
    """
    name: str
    # 4 points defining the door area (normalized 0.0-1.0)
    points: list[Point] = Field(
        default_factory=list,
        json_schema_extra={
            "options": {
                "collapsed": True,
                "disable_array_reorder": False
            }
        }
    )

    # Normal direction in degrees (0=right, 90=down, 180=left, 270=up)
    # This indicates which direction is "entering"
    # User rotates the arrow in the UI to set this
    normal_angle: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Direction of 'enter' in degrees (0=right, 90=down, 180=left, 270=up)"
    )

    # Tolerance in degrees for matching movement direction (0-180)
    # A person moving within ±tolerance of the normal is "entering"
    # A person moving within ±tolerance of the opposite direction is "exiting"
    direction_tolerance: float = Field(
        default=60.0,
        ge=0.0,
        le=90.0,
        description="Tolerance in degrees for direction matching (default 60°)"
    )

    # Triggers for this door
    triggers: list[str] = Field(default_factory=lambda: ["person"])


class ExpectedEvent(BaseModel):
    """
    Expected event for system test validation.
    
    Used in test configurations to define what events should occur
    at what times in test recordings.
    """
    event_type: str = Field(description="Event type: track_new, track_lost, zone_enter, zone_exit, door_enter, door_exit")
    timestamp_start: float = Field(ge=0.0, description="Earliest expected time in seconds")
    timestamp_end: float = Field(ge=0.0, description="Latest expected time in seconds")
    zone_name: str | None = Field(default=None, description="Zone or door name (optional)")
    track_id: int | None = Field(default=None, description="Expected track ID (optional)")


class StreamConfig(BaseModel):
    """Configuration for a single RTSP input stream."""
    name: str
    rtsp_url: str
    enabled: bool = True

    # MQTT settings for this specific stream
    mqtt_topic_suffix: str = Field(description="Suffix appended to base topic, e.g. /living_room")

    # Processing settings
    confidence_threshold: float = 0.3
    # Actions to ignore (e.g., 'standing' might be noise, 'falling' is critical)
    filtered_actions: list[str] = Field(default_factory=list)

    # Event debouncing / anti-flicker (helps when pose detection drops for a frame)
    zone_enter_confirm_frames: int = Field(default=5, ge=1)
    zone_exit_confirm_frames: int = Field(default=5, ge=1)
    zone_missing_grace_seconds: float = Field(default=1, ge=0.0)

    # New track confirmation (filters false positive detections)
    # Require N consecutive frames before emitting TRACK_NEW event
    track_new_confirm_frames: int = Field(default=5, ge=1)

    zones: list[Zone] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)

    # Expected events for system test validation (only used in test config)
    expected_events: list[ExpectedEvent] = Field(default_factory=list)

class MqttSettings(BaseModel):
    """MQTT connection settings."""
    enabled: bool = False
    broker: str = "localhost"
    port: int = 1883
    base_topic: str = "motionflow"
    username: str | None = None
    password: str | None = None
    # QoS for event messages (0=at most once, 1=at least once, 2=exactly once)
    qos: int = 0


class GeneralSettings(BaseModel):
    """Global application settings."""
    # Performance tuning
    max_fps_per_stream: int = 15

    # Pipeline selection (must match a pipeline YAML in config/)
    pipeline: str = "yolo11lpose-coco-tracker"


class AppConfig(BaseModel):
    """Root configuration object."""
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    mqtt: MqttSettings = Field(default_factory=MqttSettings)
    streams: list[StreamConfig] = Field(default_factory=list)

# --- Helper Functions ---

def load_config(path: str = "config/settings.yaml") -> AppConfig:
    """Loads the YAML configuration and validates it against the schema."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return AppConfig()
    except Exception as e:
        raise ValueError(f"Error loading config: {e}") from e

def save_config(config: AppConfig, path: str = "config/settings.yaml"):
    """Saves the configuration back to YAML."""
    # model_dump (v2) or dict (v1) - assuming Pydantic v2
    # mode='json' ensures tuples are converted to lists, avoiding !!python/tuple tags
    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode='json'), f, sort_keys=False, indent=2)
