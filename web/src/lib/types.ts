export type SessionInfo = {
  token: string;
  email: string;
};

export type Person = {
  id: string;
  user_id: string;
  name: string;
  relationship_type: string;
  notes: string | null;
  first_met: string | null;
  last_contact: string | null;
};

export type ConversationPayload = {
  personId: string;
  sourceType: "manual" | "file_upload";
  language: string;
  conversationDate?: string;
  rawContent?: string;
  file?: File | null;
};

export type RetrievedChunk = {
  chunk_id: string;
  chunk_text: string;
  chunk_index?: number;
  score: number;
  rank: number;
  conversation_id?: string | null;
};

export type BriefingResult = {
  person_id: string;
  briefing: string;
  retrieved_chunks: RetrievedChunk[];
};

export type AskResult = {
  person_id: string;
  question: string;
  answer: string;
  interaction_id: string;
  retrieved_chunks: RetrievedChunk[];
};

export type Interaction = {
  id: string;
  person_id: string;
  interaction_type: string;
  ai_advice_given: string;
  user_rating: number | null;
  created_at: string;
};

export type InteractionSummary = {
  person_id: string;
  total_interactions: number;
  rated_interactions: number;
  average_rating: number | null;
  rating_counts: Record<string, number>;
  interaction_type_counts: Record<string, number>;
};
