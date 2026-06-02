import {
  callIntersight,
  configureCredentials,
  getBaseUrl,
  isConfigured,
  trimResults,
} from "./intersight-api.js";

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handler: (args: Record<string, unknown>) => Promise<unknown>;
}

const odataParams = {
  filter: {
    type: "string",
    description:
      "OData $filter expression, e.g. \"Name eq 'my-profile'\" or \"contains(Name,'web')\".",
  },
  select: {
    type: "string",
    description: "OData $select. Comma-separated list of fields to return.",
  },
  top: {
    type: "integer",
    description: "OData $top. Max number of results to return (default 50).",
    minimum: 1,
    maximum: 1000,
  },
  skip: {
    type: "integer",
    description: "OData $skip. Number of results to skip for pagination.",
    minimum: 0,
  },
  orderby: {
    type: "string",
    description: "OData $orderby expression, e.g. \"CreateTime desc\".",
  },
} as const;

function odataQueryFromArgs(
  args: Record<string, unknown>,
): Record<string, string | number | undefined> {
  const top = args.top !== undefined ? Number(args.top) : 50;
  return {
    $filter: args.filter as string | undefined,
    $select: args.select as string | undefined,
    $top: top,
    $skip: args.skip !== undefined ? Number(args.skip) : undefined,
    $orderby: args.orderby as string | undefined,
  };
}

function listTool(opts: {
  name: string;
  description: string;
  endpoint: string;
  defaultSelect?: string[];
}): ToolDefinition {
  return {
    name: opts.name,
    description: opts.description,
    inputSchema: {
      type: "object",
      properties: { ...odataParams },
      additionalProperties: false,
    },
    handler: async (args) => {
      const result = await callIntersight({
        method: "GET",
        path: opts.endpoint,
        query: odataQueryFromArgs(args),
      });
      if (!result.ok) return result;
      // If caller asked for specific fields, respect that — otherwise trim.
      const trimmed =
        args.select === undefined
          ? trimResults(result.data, opts.defaultSelect)
          : result.data;
      return { ...result, data: trimmed };
    },
  };
}

export const tools: ToolDefinition[] = [
  {
    name: "configure_credentials",
    description:
      "Configure the Intersight v3 API Key ID and PEM private key for the current MCP session. Must be called before any Intersight tool. Credentials are held in memory only.",
    inputSchema: {
      type: "object",
      properties: {
        key_id: {
          type: "string",
          description: "Intersight API Key ID (looks like XXXX/YYYY/ZZZZ).",
        },
        pem: {
          type: "string",
          description: "Full PEM-encoded private key contents (including BEGIN/END lines).",
        },
        base_url: {
          type: "string",
          description:
            "Optional base URL. Defaults to https://intersight.com. Override for appliance deployments.",
        },
      },
      required: ["key_id", "pem"],
      additionalProperties: false,
    },
    handler: async (args) => {
      try {
        configureCredentials({
          keyId: String(args.key_id ?? ""),
          pem: String(args.pem ?? ""),
          baseUrl: String(args.base_url ?? "https://intersight.com"),
        });
        return {
          ok: true,
          message: "Credentials configured.",
          base_url: getBaseUrl(),
        };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    },
  },

  {
    name: "test_connection",
    description:
      "Verify the configured Intersight credentials by making a small probe call to /api/v1/iam/Accounts.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    handler: async () => {
      if (!isConfigured()) {
        return {
          ok: false,
          error:
            "Credentials not configured. Call configure_credentials first.",
        };
      }
      const res = await callIntersight({
        method: "GET",
        path: "/api/v1/iam/Accounts",
        query: { $top: 1 },
      });
      return res;
    },
  },

  listTool({
    name: "get_server_profiles",
    description:
      "List Intersight server profiles. WHEN: user asks about profiles, " +
      "assignments, unassigned/assigned profiles, profile templates, " +
      "config contexts, or 'which servers have profiles'. AssignedServer " +
      "is non-empty when the profile is bound to a server. PAIR WITH: " +
      "get_server_profile_by_name for drill-down on a specific named " +
      "profile. Supports OData $filter/$select/$top/$skip/$orderby.",
    endpoint: "/api/v1/server/Profiles",
    defaultSelect: [
      "Name",
      "Moid",
      "Description",
      "ConfigContext",
      "AssignedServer",
      "TargetPlatform",
      "Type",
      "ModTime",
    ],
  }),

  {
    name: "get_server_profile_by_name",
    description:
      "Get a specific server profile by exact Name. WHEN: the user names " +
      "a specific profile (e.g. 'tell me about profile X', 'show " +
      "MyProfile-01'). AVOID: don't use this to list multiple profiles — " +
      "use get_server_profiles for that.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Exact server profile Name." },
        select: odataParams.select,
      },
      required: ["name"],
      additionalProperties: false,
    },
    handler: async (args) => {
      const name = String(args.name ?? "");
      return await callIntersight({
        method: "GET",
        path: "/api/v1/server/Profiles",
        query: {
          $filter: `Name eq '${name.replace(/'/g, "''")}'`,
          $select: args.select as string | undefined,
          $top: 1,
        },
      });
    },
  },

  listTool({
    name: "get_physical_servers",
    description:
      "List all physical compute servers (blades AND rack units in one " +
      "call). DEFAULT TOOL for broad inventory questions: 'list servers', " +
      "'show me my servers', 'server inventory', 'what servers do I have', " +
      "'how many servers'. AVOID: don't separately call get_compute_blades " +
      "AND get_compute_rack_units when this single call covers both. " +
      "AVOID: don't use this for chassis-slot questions — it doesn't carry " +
      "the chassis reference; call get_chassis + get_compute_blades + " +
      "get_pci_nodes instead.",
    endpoint: "/api/v1/compute/PhysicalSummaries",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "ManagementMode",
      "OperPowerState",
      "AdminPowerState",
      "Firmware",
      "NumCpus",
      "NumThreads",
      "TotalMemory",
      "Vendor",
    ],
  }),

  listTool({
    name: "get_chassis",
    description:
      "List chassis inventory. IMPORTANT: for ANY user question that mentions " +
      "chassis ('list chassis', 'show chassis', 'chassis info', 'chassis " +
      "details', 'how many chassis', etc.) you MUST call get_compute_blades " +
      "AND get_pci_nodes in the SAME TURN as this tool — the answer needs " +
      "all three to compute slot occupancy. A chassis answer without slot " +
      "info is incomplete. Each chassis has a Moid you can join with " +
      "compute Blades' EquipmentChassis.Moid (or Chassis.Moid for older " +
      "payloads — check both). NumSlots is often empty for X-Series chassis; " +
      "use known capacities by model when missing (UCSX-9508 = 8, " +
      "UCSB-5108-AC2 = 8).",
    // Intersight uses the irregular plural `Chasses` for this endpoint (verified
    // against CiscoDevNet/intersight-python's equipment_api.py). The MO type is
    // still `equipment.Chassis` and the field name on Blades is still `Chassis.Moid`.
    endpoint: "/api/v1/equipment/Chasses",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "Vendor",
      "OperState",
      "ManagementMode",
      "NumSlots",
    ],
  }),

  listTool({
    name: "get_compute_blades",
    description:
      "List blade servers. WHEN: user asks specifically about blades, " +
      "blade hardware, blade slots within a chassis, or 'which blades are " +
      "in chassis X'. MUST PAIR WITH get_chassis AND get_pci_nodes for " +
      "any chassis-slot, capacity, or 'free slots' question — slot math " +
      "needs all three. AVOID: for generic 'list servers' or 'server " +
      "inventory' questions without a blade-specific angle, use " +
      "get_physical_servers (which covers blades + rack units in one " +
      "call). Each blade has SlotId (slot in parent chassis) and a chassis " +
      "reference. Modern Intersight uses EquipmentChassis (canonical); " +
      "older payloads use Chassis — check both.",
    endpoint: "/api/v1/compute/Blades",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "SlotId",
      // Both possible chassis-ref field names. Newer Intersight uses
      // EquipmentChassis; some older payloads still use Chassis. Including
      // both in defaultSelect means trimResults preserves whichever is
      // populated — without it, the blade -> chassis join silently breaks.
      "EquipmentChassis",
      "Chassis",
      "OperState",
      "OperPowerState",
      "AdminPowerState",
      "TotalMemory",
      "NumCpus",
      "NumThreads",
    ],
  }),

  listTool({
    name: "get_compute_rack_units",
    description:
      "List rack-mount servers (standalone or UCS-managed rack units). " +
      "WHEN: user asks specifically about rack servers, rack hardware, " +
      "rack-mount inventory, or 'C-series servers'. AVOID: for generic " +
      "'list servers' or 'server inventory' questions, use " +
      "get_physical_servers (single call for blades + racks).",
    endpoint: "/api/v1/compute/RackUnits",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "OperState",
      "OperPowerState",
      "AdminPowerState",
      "TotalMemory",
      "NumCpus",
      "NumThreads",
    ],
  }),

  listTool({
    name: "get_pci_nodes",
    description:
      "List PCIe nodes (pci.Node MOs, e.g., UCSX-440P GPU/storage nodes in " +
      "X-Series chassis). REQUIRED for ANY chassis-related question — " +
      "calling get_chassis or get_compute_blades without also calling " +
      "get_pci_nodes produces WRONG slot occupancy numbers, because " +
      "PCIe nodes occupy chassis slots too. If you've already called " +
      "get_chassis OR get_compute_blades and have NOT called this yet, " +
      "you must call it now before answering. WHEN (also): user asks " +
      "about PCIe nodes, GPU nodes, X440p, or expansion nodes directly. " +
      "PCIe nodes do NOT reference a chassis directly; they reference " +
      "their paired blade via ComputeBlade (Parent as fallback). Two-hop " +
      "join: pci.Node -> compute.Blade -> equipment.Chassis. Used slots " +
      "in a chassis = blades + PCIe nodes whose paired blade is in that " +
      "chassis.",
    endpoint: "/api/v1/pci/Nodes",
    defaultSelect: [
      "Name",
      "Moid",
      "Model",
      "SlotId",
      "ComputeBlade",
      "Parent",
      "ObjectType",
    ],
  }),

  listTool({
    name: "get_fabric_interconnects",
    description:
      "List fabric interconnects (network elements). WHEN: user asks " +
      "about fabric interconnects, FIs, 'FI-A' / 'FI-B', fabric switches, " +
      "or the upstream fabric layer. The admin-configured Name is often " +
      "empty — when it is, identify the FI by Switchid (always 'A' or 'B', " +
      "commonly referred to as 'FI-A' / 'FI-B'), Hostname, or Dn (e.g., " +
      "'sys/switch-A'), in that order of preference.",
    endpoint: "/api/v1/network/Elements",
    defaultSelect: [
      "Name",
      "Hostname",
      "Dn",
      "Moid",
      "Serial",
      "Model",
      "Vendor",
      "OperState",
      "ManagementMode",
      "OutOfBandIpAddress",
      "Switchid",
      "Version",
    ],
  }),

  listTool({
    name: "get_alarms",
    description:
      "List individual active alarms with details. WHEN: user wants " +
      "alarm DETAILS — names, descriptions, affected MOs, timestamps, " +
      "acknowledge state — or wants to filter by severity (e.g., 'show me " +
      "all critical alarms', 'what alarms fired today'). AVOID: for " +
      "count-only questions like 'how many alarms?' or 'is anything " +
      "broken?', use get_alarm_summary instead — it's much cheaper. " +
      "Defaults to most recent first; filter with $filter, e.g. " +
      "\"Severity eq 'Critical'\".",
    endpoint: "/api/v1/cond/Alarms",
    defaultSelect: [
      "Name",
      "Moid",
      "Severity",
      "Description",
      "Code",
      "AffectedMoDisplayName",
      "AffectedObjectType",
      "CreationTime",
      "LastTransitionTime",
      "Acknowledge",
    ],
  }),

  listTool({
    name: "get_hcl_status",
    description:
      "List HCL (Hardware Compatibility List) compatibility statuses " +
      "per server. WHEN: user asks about HCL, compatibility, supported " +
      "hardware, 'is my hardware supported', or compliance against " +
      "Cisco's HCL. Status field is the main indicator (Validated / " +
      "Incomplete / Not-Validated / Not-Listed).",
    endpoint: "/api/v1/cond/HclStatuses",
    defaultSelect: [
      "Moid",
      "Status",
      "Reason",
      "ServerReason",
      "InvalidReasons",
      "HardwareStatus",
      "SoftwareStatus",
      "ManagedObject",
    ],
  }),

  listTool({
    name: "get_running_firmware",
    description:
      "List running firmware versions across managed components. WHEN: " +
      "user asks about firmware, software versions, 'what firmware is " +
      "running where', or fleet-wide version compliance. Each entry ties " +
      "a Component (hardware unit) to its current Version. For 'is my " +
      "fleet running X' or 'show me anything not on Y', filter on " +
      "Version or Component.",
    endpoint: "/api/v1/firmware/RunningFirmwares",
    defaultSelect: [
      "Moid",
      "Component",
      "Version",
      "PackageVersion",
      "Type",
      "Vendor",
      "Model",
    ],
  }),

  {
    name: "get_alarm_summary",
    description:
      "Fleet-wide alarm count rolled up by severity (Critical, Warning, " +
      "Info, Cleared). WHEN: questions like 'how many critical alarms?', " +
      "'is anything broken?', 'health overview', 'any alerts?'. PAIR WITH: " +
      "get_alarms if the user then asks for specifics. AVOID: don't use " +
      "get_alarms to count — this tool is much cheaper for counts.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    handler: async () => {
      // Intersight doesn't expose a dedicated /cond/AlarmSummary endpoint;
      // we build the summary using OData $apply=groupby on /cond/Alarms.
      return await callIntersight({
        method: "GET",
        path: "/api/v1/cond/Alarms",
        query: {
          $apply: "groupby((Severity),aggregate($count as Count))",
        },
      });
    },
  },

  listTool({
    name: "get_organizations",
    description:
      "List Intersight organizations. Use this when the user asks about a specific " +
      "tenant/org, or to scope a follow-up query (most resources have an Organization " +
      "reference you can $filter on).",
    endpoint: "/api/v1/organization/Organizations",
    defaultSelect: ["Name", "Moid", "Description", "AccountMoid"],
  }),

  listTool({
    name: "get_advisories",
    description:
      "List active advisory instances (PSIRTs, field notices) affecting your fleet. " +
      "Each instance has a State (active/cleared), an AffectedObject reference, and a " +
      "Definition reference. To see advisory titles/descriptions, follow up with " +
      "generic_api_call to /api/v1/tam/AdvisoryDefinitions for the relevant Moids, or " +
      "pass select='*' here for the full record.",
    endpoint: "/api/v1/tam/AdvisoryInstances",
    defaultSelect: ["Moid", "State", "AffectedObject", "Definition", "LastVisibleTime"],
  }),

  listTool({
    name: "get_contracts",
    description:
      "List device contract / service-coverage information. Use this for questions " +
      "like 'what contracts are expiring?' or 'show me servers without coverage'. " +
      "Pass orderby='ServiceEndDate asc' to see soonest-to-expire first; filter on " +
      "ContractStatus eq 'Expired' or 'Active' to narrow.",
    endpoint: "/api/v1/asset/DeviceContractInformations",
    defaultSelect: [
      "Moid",
      "ContractStatus",
      "ContractStatusReason",
      "ServiceEndDate",
      "ServiceLevel",
      "ServiceSku",
      "ProductId",
      "DeviceId",
      "DeviceType",
      "Source",
    ],
  }),

  {
    name: "generic_api_call",
    description:
      "ESCAPE HATCH — last resort only. STOP and check whether a " +
      "dedicated tool fits before calling this. For ANY of these " +
      "endpoints, you MUST use the dedicated tool instead — calling " +
      "generic_api_call for these is a BUG: " +
      "/api/v1/equipment/Chasses -> get_chassis; " +
      "/api/v1/compute/Blades -> get_compute_blades; " +
      "/api/v1/compute/RackUnits -> get_compute_rack_units; " +
      "/api/v1/compute/PhysicalSummaries -> get_physical_servers; " +
      "/api/v1/pci/Nodes -> get_pci_nodes; " +
      "/api/v1/network/Elements -> get_fabric_interconnects; " +
      "/api/v1/server/Profiles -> get_server_profiles (or _by_name); " +
      "/api/v1/cond/Alarms -> get_alarms or get_alarm_summary; " +
      "/api/v1/cond/HclStatuses -> get_hcl_status; " +
      "/api/v1/firmware/RunningFirmwares -> get_running_firmware; " +
      "/api/v1/organization/Organizations -> get_organizations; " +
      "/api/v1/tam/AdvisoryInstances -> get_advisories; " +
      "/api/v1/asset/DeviceContractInformations -> get_contracts. " +
      "WHEN to ACTUALLY use this: the user asks about endpoints with " +
      "NO dedicated tool — storage, virtualization, workflow, OS " +
      "install, kubernetes, software-repository, etc. Dedicated tools " +
      "return curated/trimmed responses; this returns the full payload " +
      "and burns context. NEVER use POST / PATCH / DELETE / PUT unless " +
      "the user explicitly asks to modify state. Supports any method, " +
      "any path under /api/v1/, query parameters, optional JSON body.",
    inputSchema: {
      type: "object",
      properties: {
        method: {
          type: "string",
          enum: ["GET", "POST", "PATCH", "DELETE", "PUT"],
          description: "HTTP method.",
        },
        path: {
          type: "string",
          description:
            "Path under the Intersight base URL, e.g. /api/v1/asset/DeviceRegistrations.",
        },
        query: {
          type: "object",
          description:
            "Query parameters. For OData, use keys like $filter, $select, $top, $skip, $orderby.",
          additionalProperties: { type: ["string", "number", "boolean"] },
        },
        body: {
          description: "Optional JSON body for POST/PATCH/PUT.",
        },
      },
      required: ["method", "path"],
      additionalProperties: false,
    },
    handler: async (args) => {
      return await callIntersight({
        method: args.method as "GET" | "POST" | "PATCH" | "DELETE" | "PUT",
        path: String(args.path),
        query: args.query as Record<string, string | number | boolean> | undefined,
        body: args.body,
      });
    },
  },
];

// ---------------------------------------------------------------- list_tools
//
// Pushed onto `tools` after the array is declared so the handler can close
// over `tools` (referencing it inside an array literal at module top-level
// would hit a TDZ). Exposed to the model so it can answer "what tools do
// you have?" by calling a function rather than enumerating from memory —
// smaller models tend to truncate self-described tool lists, and the
// missing entries can mislead users about what the app can do.

const HIDDEN_FROM_LISTING = new Set([
  // configure_credentials is host-only (the Streamlit sidebar wires it).
  // It's already hidden from the model in orchestrator.py's HIDDEN_TOOLS;
  // exclude it here too so list_tools output matches the model's actual
  // tool surface.
  "configure_credentials",
]);

tools.push({
  name: "list_tools",
  description:
    "Return the complete list of MCP tools available on this server, with " +
    "each tool's name and a SHORT one-sentence description. ALWAYS call " +
    "this tool when the user asks what tools, commands, capabilities, " +
    "MCPs, or functions you have — the result is the authoritative list. " +
    "Do not summarize from memory; always call this tool for these " +
    "questions. Render the result as a markdown table with TWO columns: " +
    "'Tool' and 'What it does'. Include every entry from the result — " +
    "do not summarize, group, omit, or invent rows. Begin the table on a " +
    "new line and produce a proper markdown separator row (|---|---|) " +
    "right after the header.",
  inputSchema: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
  handler: async () => {
    // Trim each description to its first sentence. The full descriptions
    // are dense routing guidance (WHEN/PAIR WITH/AVOID), which is great
    // for the model's tool-selection schema but verbose to render as a
    // user-facing table. Smaller models (e.g. mistral-small) sometimes
    // write the header row and bail out when asked to render 6-8 KB of
    // mixed-format prose into 16 rows. First-sentence descriptions keep
    // the result compact (~30-100 chars each) so the table actually gets
    // produced end-to-end.
    const firstSentence = (desc: string): string => {
      // Split on the first period followed by whitespace or end-of-string.
      // Falls back to a generous char-cap if no sentence boundary is found.
      const m = desc.match(/^[^.]+\./);
      return (m ? m[0] : desc).trim().slice(0, 140);
    };
    const visible = tools.filter((t) => !HIDDEN_FROM_LISTING.has(t.name));
    return {
      ok: true,
      count: visible.length,
      tools: visible.map((t) => ({
        name: t.name,
        description: firstSentence(t.description),
      })),
    };
  },
});

export const toolMap = new Map(tools.map((t) => [t.name, t]));
