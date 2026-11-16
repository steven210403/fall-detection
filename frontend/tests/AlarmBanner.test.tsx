import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AlarmBanner } from "../src/components/AlarmBanner";
import { fallEvent } from "./fixtures";

describe("AlarmBanner", () => {
  it("renders nothing without an active alarm", () => {
    const { container } = render(
      <AlarmBanner alarms={[]} muted={false} onToggleMute={vi.fn()} onAcknowledge={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("announces a fall with text, not colour alone", () => {
    render(<AlarmBanner alarms={[fallEvent()]} muted onToggleMute={vi.fn()} onAcknowledge={vi.fn()} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("PERINGATAN: terdeteksi orang jatuh");
    expect(alert).toHaveTextContent("Orang #3");
    expect(alert).toHaveTextContent("respons 3.0 dtk");
  });

  it("acknowledges one alarm", () => {
    const onAcknowledge = vi.fn();
    render(<AlarmBanner alarms={[fallEvent()]} muted onToggleMute={vi.fn()} onAcknowledge={onAcknowledge} />);
    fireEvent.click(screen.getByRole("button", { name: "Konfirmasi" }));
    expect(onAcknowledge).toHaveBeenCalledWith("e1");
  });

  it("shows an acknowledged alarm without the button", () => {
    render(
      <AlarmBanner
        alarms={[fallEvent({ acknowledged_utc: 1_700_000_010 })]}
        muted
        onToggleMute={vi.fn()}
        onAcknowledge={vi.fn()}
      />,
    );
    expect(screen.queryByRole("button", { name: "Konfirmasi" })).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("dikonfirmasi");
  });

  it("counts several people", () => {
    render(
      <AlarmBanner
        alarms={[fallEvent(), fallEvent({ event_id: "e2", display_id: 4 })]}
        muted
        onToggleMute={vi.fn()}
        onAcknowledge={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("2 orang jatuh");
  });

  it("shows a failed acknowledgement on the banner", () => {
    render(
      <AlarmBanner
        alarms={[fallEvent()]}
        muted
        onToggleMute={vi.fn()}
        onAcknowledge={vi.fn()}
        error="server tidak menjawab"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Konfirmasi gagal: server tidak menjawab");
  });

  it("offers to turn a muted alarm back on", () => {
    render(<AlarmBanner alarms={[fallEvent()]} muted onToggleMute={vi.fn()} onAcknowledge={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Nyalakan suara" })).toBeInTheDocument();
  });

  it("toggles the sound", () => {
    const onToggleMute = vi.fn();
    render(<AlarmBanner alarms={[fallEvent()]} muted={false} onToggleMute={onToggleMute} onAcknowledge={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /Matikan suara/ }));
    expect(onToggleMute).toHaveBeenCalled();
  });
});
