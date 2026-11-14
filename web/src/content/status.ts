/**
 * The only status vocabulary the site may use. A model, product or report
 * moves forward only when that has actually happened.
 */
export const STATUSES = ["planned", "in-development", "training", "trained-internal", "released"] as const;
export type Status = (typeof STATUSES)[number];

export const STATUS_INFO: Record<Status, { label: string; description: string }> = {
  planned: { label: "Planned", description: "On the roadmap. Work has not started." },
  "in-development": { label: "In development", description: "Being built. Not trained or available yet." },
  training: { label: "Training", description: "A training run is in progress." },
  "trained-internal": { label: "Trained (internal)", description: "Trained and evaluated internally. Not publicly available." },
  released: { label: "Released", description: "Publicly available." },
};

/** Status for pipeline components (code, not models). */
export type ComponentStatus = "working" | "planned";
export const COMPONENT_STATUS_INFO: Record<ComponentStatus, { label: string }> = {
  working: { label: "Built & tested" },
  planned: { label: "Planned" },
};
