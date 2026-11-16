import { useId } from "react";
import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  /** One line under the title saying what the panel shows. */
  description?: ReactNode;
  /** Controls or a counter, right-aligned in the header. */
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * The dashboard's region: a titled, hairline-ruled block with square corners and no shadow.
 *
 * Only the alarm banner lifts off the page; every panel sits flat on it. The title is set in
 * the mono face, the dashboard's schematic voice.
 */
export function Panel({ title, description, actions, children, className = "" }: PanelProps) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className={`border border-line bg-surface ${className}`}>
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3">
        <div className="min-w-0">
          <h2 id={headingId} className="font-mono text-sm font-semibold text-fg">
            {title}
          </h2>
          {description ? <p className="mt-1 text-sm text-fg-muted">{description}</p> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}
