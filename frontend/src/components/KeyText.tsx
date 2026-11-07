import { Fragment } from "react";

/** A raw tensor key as plain text, with line-break opportunities after "." and "/" so long names wrap at
 * segment boundaries. <wbr> adds no characters: textContent and copied text equal the key exactly. */
export function KeyText({ value }: { value: string }) {
  const parts = value.split(/(?<=[./])/);
  return (
    <>
      {parts.map((p, i) => (
        <Fragment key={i}>
          {p}
          {i < parts.length - 1 ? <wbr /> : null}
        </Fragment>
      ))}
    </>
  );
}
