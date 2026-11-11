export type Section = "workspace" | "explore" | "compare" | "adapters" | "settings";

const ITEMS: { id: Section; label: string; planned?: string }[] = [
  { id: "workspace", label: "Workspace" },
  { id: "explore", label: "Explore" },
  { id: "compare", label: "Compare" },
  { id: "adapters", label: "Adapters", planned: "planned for a later milestone" },
  { id: "settings", label: "Settings" },
];

export function SideNav({ section, onSection }: { section: Section; onSection: (s: Section) => void }) {
  return (
    <nav className="sidenav" aria-label="Main">
      <ul>
        {ITEMS.map((it) => (
          <li key={it.id}>
            {it.planned ? (
              <span className="nav-item nav-disabled" aria-disabled="true" title={`${it.label}: ${it.planned}`}>
                {it.label}
                <span className="nav-tag">planned</span>
              </span>
            ) : (
              <button
                type="button"
                className="nav-item"
                aria-current={section === it.id ? "page" : undefined}
                onClick={() => onSection(it.id)}
              >
                {it.label}
              </button>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}
