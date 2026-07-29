/**
 * Types mirroring the backend domain contract (backend/app/domain/models.py).
 *
 * Kept in sync by a contract test: `backend/tests/integration/test_openapi_contract.py`
 * asserts that every field the UI reads is present in the generated OpenAPI
 * schema, so a rename on either side fails CI rather than the dashboard.
 */

export type LeadBand = 'hot' | 'warm' | 'cold' | 'disqualified';

export type AnalysisStatus = 'pending' | 'running' | 'completed' | 'failed';

export type SpeakerRole = 'rep' | 'prospect' | 'unknown';

export type SignalType =
  | 'budget'
  | 'authority'
  | 'need'
  | 'timeline'
  | 'urgency'
  | 'competitor'
  | 'enthusiasm'
  | 'next_step'
  | 'technical_fit'
  | 'social_proof';

export type ObjectionCategory =
  | 'price'
  | 'timing'
  | 'authority'
  | 'competitor'
  | 'need'
  | 'trust'
  | 'complexity';

export type CallOutcomeLabel =
  | 'meeting_booked'
  | 'follow_up_agreed'
  | 'information_requested'
  | 'not_interested'
  | 'not_a_fit'
  | 'gatekeeper_only'
  | 'no_clear_outcome';

export type ActionPriority = 'p0' | 'p1' | 'p2';

export interface Evidence {
  segment_index: number;
  quote: string;
  speaker: string;
  role: SpeakerRole;
  start_s: number | null;
  matched_phrase: string | null;
}

export interface SentimentPoint {
  position: number;
  at_s: number | null;
  score: number;
  segment_index: number;
}

export interface SentimentAnalysis {
  overall: number;
  prospect: number | null;
  rep: number | null;
  timeline: SentimentPoint[];
  trend: number;
  volatility: number;
  confidence: number;
}

export interface Signal {
  type: SignalType;
  label: string;
  strength: number;
  evidence: Evidence[];
}

export interface Objection {
  category: ObjectionCategory;
  label: string;
  severity: number;
  evidence: Evidence[];
  suggested_response: string;
  addressed: boolean;
}

export interface ScoreComponent {
  name: string;
  raw: number;
  weight: number;
  contribution: number;
  detail: string;
}

export interface LeadScore {
  score: number;
  band: LeadBand;
  confidence: number;
  components: ScoreComponent[];
  penalties: ScoreComponent[];
  rationale: string[];
  risks: string[];
}

export interface CallOutcome {
  label: CallOutcomeLabel;
  confidence: number;
  evidence: Evidence[];
}

export interface NextAction {
  action: string;
  priority: ActionPriority;
  rationale: string;
  due_within_hours: number;
}

export interface CallSummary {
  headline: string;
  bullets: string[];
  key_quotes: Evidence[];
}

export interface TalkMetrics {
  rep_talk_ratio: number;
  prospect_talk_ratio: number;
  longest_rep_monologue_words: number;
  question_count_rep: number;
  question_count_prospect: number;
  turn_count: number;
  segments_analysed: number;
}

export interface AnalysisMetadata {
  engine_version: string;
  schema_version: number;
  analyzers: Record<string, string>;
  processing_ms: number;
  word_count: number;
  duration_s: number | null;
  duration_is_estimated: boolean;
  warnings: string[];
}

export interface CallAnalysis {
  call_id: string;
  analyzed_at: string;
  summary: CallSummary;
  sentiment: SentimentAnalysis;
  signals: Signal[];
  objections: Objection[];
  lead_score: LeadScore;
  outcome: CallOutcome;
  next_actions: NextAction[];
  talk_metrics: TalkMetrics;
  topics: string[];
  metadata: AnalysisMetadata;
}

export interface TranscriptSegment {
  index: number;
  speaker: string;
  role: SpeakerRole;
  text: string;
  start_s: number | null;
  end_s: number | null;
  duration_s: number | null;
  word_count: number;
}

export interface Transcript {
  source_format: string;
  segments: TranscriptSegment[];
  language: string;
  text: string;
  word_count: number;
  duration_s: number | null;
}

export interface CallSummaryRow {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  status: AnalysisStatus;
  source_format: string;
  lead_score: number | null;
  lead_band: LeadBand | null;
  outcome: CallOutcomeLabel | null;
  sentiment: number | null;
  duration_s: number | null;
  word_count: number;
  owner: string | null;
  tags: string[];
  error: string | null;
}

export interface CallDetail extends CallSummaryRow {
  source_uri: string | null;
  analysis: CallAnalysis | null;
  transcript: Transcript | null;
}

export interface PageMeta {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface Paginated<T> {
  items: T[];
  pagination: PageMeta;
}

export interface BandBreakdown {
  hot: number;
  warm: number;
  cold: number;
  disqualified: number;
}

export interface TrendPoint {
  date: string;
  calls: number;
  avg_lead_score: number | null;
  avg_sentiment: number | null;
}

export interface PipelineAnalytics {
  total_calls: number;
  analysed_calls: number;
  failed_calls: number;
  avg_lead_score: number | null;
  avg_sentiment: number | null;
  band_breakdown: BandBreakdown;
  outcome_breakdown: Record<string, number>;
  top_objections: { category: string; count: number; unaddressed: number }[];
  top_signals: { type: string; count: number }[];
  top_risks: { risk: string; count: number }[];
  trend: TrendPoint[];
  generated_at: string;
}

export interface CallFilters {
  search?: string;
  band?: LeadBand[];
  owner?: string;
  min_score?: number;
  sort_by?: string;
  sort_desc?: boolean;
  page?: number;
  page_size?: number;
}
