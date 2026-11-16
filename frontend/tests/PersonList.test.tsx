import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PersonList } from "../src/components/PersonList";
import { track } from "./fixtures";

describe("PersonList", () => {
  it("says when nobody is tracked", () => {
    render(<PersonList tracks={[]} />);
    expect(screen.getByRole("status")).toHaveTextContent("Tidak ada orang yang terlacak");
  });

  it("shows the state and the three measurements", () => {
    render(<PersonList tracks={[track()]} />);
    const item = screen.getByRole("listitem");
    expect(item).toHaveTextContent("Orang #3");
    expect(item).toHaveTextContent("Normal");
    expect(item).toHaveTextContent("84°");
    expect(item).toHaveTextContent("0.31");
    expect(item).toHaveTextContent("0.02");
  });

  it("shows the long-lie countdown during impact", () => {
    const base = track();
    render(
      <PersonList
        tracks={[
          track({
            fall: base.fall && { ...base.fall, state: "impact", long_lie_remaining_s: 1.4 },
          }),
        ]}
      />,
    );
    expect(screen.getByRole("listitem")).toHaveTextContent("Tergeletak");
    expect(screen.getByRole("listitem")).toHaveTextContent("Alarm dalam 1.4 dtk");
  });

  it("shows a fallen person with a glyph and text", () => {
    const base = track();
    render(<PersonList tracks={[track({ fall: base.fall && { ...base.fall, state: "fallen" } })]} />);
    expect(screen.getByRole("listitem")).toHaveTextContent("Jatuh: alarm aktif");
  });

  it("shows an unmeasured value as a dash, never zero", () => {
    const base = track();
    render(
      <PersonList
        tracks={[
          track({
            fall: base.fall && {
              ...base.fall,
              features: { ...base.fall.features, torso_angle_deg: null, vy_norm: null },
            },
          }),
        ]}
      />,
    );
    const item = screen.getByRole("listitem");
    expect(item).toHaveTextContent("Sudut torso—");
    expect(item).toHaveTextContent("Vy (tubuh/dtk)—");
  });

  it("does not call a disconnected feed an empty room", () => {
    render(<PersonList tracks={[]} feedLive={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("live feed terputus");
    expect(screen.getByRole("status")).not.toHaveTextContent("Tidak ada orang");
  });

  it("marks the cards as the last data while the feed is down", () => {
    render(<PersonList tracks={[track()]} feedLive={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("data terakhir");
    expect(screen.getByRole("listitem")).toHaveTextContent("Orang #3");
  });

  it("tells the operator to start a stopped camera", () => {
    render(<PersonList tracks={[]} cameraRunning={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("Mulai kamera");
  });

  it("shows unmeasured keypoints as a dash, never zero", () => {
    render(<PersonList tracks={[track({ fall: null })]} />);
    expect(screen.getByRole("listitem")).toHaveTextContent("— dari 17");
  });

  it("treats a person without a decision as not measured", () => {
    render(<PersonList tracks={[track({ fall: null })]} />);
    expect(screen.getByRole("listitem")).toHaveTextContent("Tidak terukur");
  });
});
