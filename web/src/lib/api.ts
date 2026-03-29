import type {
  AskResult,
  BriefingResult,
  ConversationPayload,
  Interaction,
  InteractionSummary,
  Person,
} from "@/lib/types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
)
  .trim()
  .replace(/\/$/, "");

type RequestOptions = {
  method?: string;
  token?: string;
  body?: unknown;
  formData?: FormData;
};

type AuthPayload = {
  email: string;
  password: string;
};

type UserResponse = {
  id: string;
  email: string;
};

type PersonPayload = {
  name: string;
  relationship_type: string;
  notes?: string;
  first_met?: string;
  last_contact?: string;
};

function readErrorDetail(payload: unknown): string | null {
  if (typeof payload === "string") {
    return payload;
  }

  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = payload.detail;
    if (typeof detail === "string") {
      return detail;
    }
  }

  return null;
}

async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const headers = new Headers();
  let body: BodyInit | undefined;

  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }

  if (options.formData) {
    body = options.formData;
  } else if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers,
    body,
  });

  if (!response.ok) {
    let message = "request failed";
    try {
      const payload = (await response.json()) as unknown;
      message = readErrorDetail(payload) ?? message;
    } catch {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }

    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function registerUser(payload: AuthPayload): Promise<UserResponse> {
  return apiRequest<UserResponse>("/api/users/register", {
    method: "POST",
    body: payload,
  });
}

export async function loginUser(
  payload: AuthPayload,
): Promise<{ access_token: string; token_type: string }> {
  return apiRequest("/api/users/login", {
    method: "POST",
    body: payload,
  });
}

export async function listPersons(token: string): Promise<Person[]> {
  return apiRequest<Person[]>("/api/persons", { token });
}

export async function createPerson(
  token: string,
  payload: PersonPayload,
): Promise<Person> {
  return apiRequest<Person>("/api/persons", {
    method: "POST",
    token,
    body: payload,
  });
}

export async function getPerson(token: string, personId: string): Promise<Person> {
  return apiRequest<Person>(`/api/persons/${personId}`, { token });
}

export async function uploadConversation(
  token: string,
  payload: ConversationPayload,
): Promise<void> {
  const formData = new FormData();
  formData.set("person_id", payload.personId);
  formData.set("source_type", payload.sourceType);
  formData.set("language", payload.language);

  if (payload.conversationDate) {
    formData.set("conversation_date", payload.conversationDate);
  }
  if (payload.rawContent) {
    formData.set("raw_content", payload.rawContent);
  }
  if (payload.file) {
    formData.set("file", payload.file);
  }

  await apiRequest("/api/conversations", {
    method: "POST",
    token,
    formData,
  });
}

export async function askQuestion(
  token: string,
  payload: {
    person_id: string;
    question: string;
    top_k?: number;
  },
): Promise<AskResult> {
  return apiRequest<AskResult>("/api/ask", {
    method: "POST",
    token,
    body: payload,
  });
}

export async function getBriefing(
  token: string,
  personId: string,
  topK = 5,
): Promise<BriefingResult> {
  return apiRequest<BriefingResult>(
    `/api/persons/${personId}/briefing?top_k=${topK}`,
    { token },
  );
}

export async function getInteractions(
  token: string,
  personId: string,
  limit = 20,
): Promise<Interaction[]> {
  return apiRequest<Interaction[]>(
    `/api/persons/${personId}/interactions?limit=${limit}`,
    { token },
  );
}

export async function getInteractionSummary(
  token: string,
  personId: string,
  limit = 50,
): Promise<InteractionSummary> {
  return apiRequest<InteractionSummary>(
    `/api/persons/${personId}/interactions/summary?limit=${limit}`,
    { token },
  );
}

export async function rateInteraction(
  token: string,
  personId: string,
  interactionId: string,
  userRating: number,
): Promise<Interaction> {
  return apiRequest<Interaction>(
    `/api/persons/${personId}/interactions/${interactionId}/rating`,
    {
      method: "PATCH",
      token,
      body: { user_rating: userRating },
    },
  );
}
