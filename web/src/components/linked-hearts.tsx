import type { SVGProps } from "react";

export function LinkedHearts(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 360 360"
      fill="none"
      stroke="currentColor"
      strokeWidth="7"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <circle cx="180" cy="180" r="128" opacity="0.14" />
      <circle cx="180" cy="180" r="92" opacity="0.18" />
      <path
        d="M82 132c0-30 21-52 49-52 18 0 33 10 41 24 8-14 23-24 41-24 28 0 49 22 49 52 0 33-28 56-56 80l-34 29-34-29c-28-24-56-47-56-80Z"
        opacity="0.84"
      />
      <path
        d="M147 181c9-13 21-20 33-20 13 0 25 7 34 21"
        opacity="0.9"
      />
      <path
        d="M88 248c24-8 46-12 68-12 23 0 43 4 63 12"
        opacity="0.26"
      />
      <path
        d="M228 112c18-7 38-5 52 5 14 10 23 27 24 45 2 29-15 48-34 67l-23 22"
        opacity="0.56"
      />
      <path
        d="M121 113c-18-7-38-5-52 5-14 10-23 27-24 45-2 29 15 48 34 67l23 22"
        opacity="0.56"
      />
      <path d="M248 167c21 5 35 13 48 28" opacity="0.66" />
      <circle cx="304" cy="202" r="10" fill="currentColor" stroke="none" opacity="0.18" />
      <path d="M112 167c-21 5-35 13-48 28" opacity="0.66" />
      <circle cx="56" cy="202" r="10" fill="currentColor" stroke="none" opacity="0.18" />
    </svg>
  );
}
