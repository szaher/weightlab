import type { ReactNode } from "react";

interface Props {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  help: ReactNode;
  error?: string | undefined;
  required?: boolean;
  placeholder?: string;
  maxLength: number;
  mono?: boolean;
}

export function PathField({ id, label, value, onChange, help, error, required, placeholder, maxLength, mono = true }: Props) {
  const helpId = `${id}-help`;
  const errId = `${id}-err`;
  return (
    <div className="field">
      <label htmlFor={id} className="label">
        {label}
        {required ? <span className="req"> (required)</span> : <span className="muted"> (optional)</span>}
      </label>
      <input
        id={id}
        className={mono ? "input mono" : "input"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        autoComplete="off"
        autoCapitalize="off"
        maxLength={maxLength + 1}
        placeholder={placeholder}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${errId} ${helpId}` : helpId}
        required={required}
      />
      {error ? (
        <p id={errId} className="field-error">
          {error}
        </p>
      ) : null}
      <p id={helpId} className="help">
        {help}
      </p>
    </div>
  );
}
