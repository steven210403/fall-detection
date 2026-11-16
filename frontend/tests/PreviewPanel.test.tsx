import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PreviewPanel } from "../src/components/PreviewPanel";

describe("PreviewPanel", () => {
  const baseProps = {
    cameraId: "c1",
    previewUrl: "/api/v1/cameras/c1/preview.mjpeg",
    snapshotUrl: "/api/v1/cameras/c1/snapshot.jpg",
    connection: "live" as const,
    cameraState: "live",
    frameSeq: 120,
    previewBytes: 102400,
    hasFrame: true,
    onStart: vi.fn(),
    onStop: vi.fn(),
  };

  it("renders the MJPEG stream exactly as delivered", () => {
    // §18.1 forbids overlaying newer metadata on an older frame, so the client renders the
    // image as the backend drew it and puts the labels in the track list instead.
    render(<PreviewPanel {...baseProps} />);
    expect(screen.getByAltText(/Pratinjau kamera beranotasi/)).toHaveAttribute(
      "src",
      "/api/v1/cameras/c1/preview.mjpeg",
    );
  });

  it("shows the camera state and the live feed state in words", () => {
    // A frozen preview looks identical whether the camera stopped or the socket dropped, and
    // those need different responses.
    render(<PreviewPanel {...baseProps} cameraState="reconnecting" connection="reconnecting" />);
    expect(screen.getByRole("status")).toHaveTextContent("Kamera: menyambung ulang");
    expect(screen.getByRole("status")).toHaveTextContent("Live feed: menyambung ulang");
  });

  it("explains an empty preview instead of showing a broken image", () => {
    render(<PreviewPanel {...baseProps} hasFrame={false} frameSeq={null} />);
    expect(screen.queryByAltText(/Pratinjau kamera beranotasi/)).not.toBeInTheDocument();
    expect(screen.getByText(/Belum ada frame/)).toBeInTheDocument();
  });

  it("tells a stopped camera's operator what to press", () => {
    render(<PreviewPanel {...baseProps} cameraState="idle" hasFrame={false} frameSeq={null} />);
    expect(screen.getByText(/Tekan Mulai kamera/)).toBeInTheDocument();
  });

  it("disables start while running and stop while stopped", () => {
    const { rerender } = render(<PreviewPanel {...baseProps} cameraState="live" />);
    expect(screen.getByRole("button", { name: /Mulai kamera/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Hentikan kamera/ })).toBeEnabled();

    rerender(<PreviewPanel {...baseProps} cameraState="idle" />);
    expect(screen.getByRole("button", { name: /Mulai kamera/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Hentikan kamera/ })).toBeDisabled();
  });

  it("calls the handlers", async () => {
    const onStart = vi.fn();
    const onStop = vi.fn();
    const { rerender } = render(
      <PreviewPanel {...baseProps} cameraState="idle" onStart={onStart} onStop={onStop} />,
    );
    screen.getByRole("button", { name: /Mulai kamera/ }).click();
    expect(onStart).toHaveBeenCalledOnce();

    rerender(<PreviewPanel {...baseProps} cameraState="live" onStart={onStart} onStop={onStop} />);
    screen.getByRole("button", { name: /Hentikan kamera/ }).click();
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("surfaces an error", () => {
    render(<PreviewPanel {...baseProps} error="the source is not configured" />);
    expect(screen.getByRole("alert")).toHaveTextContent("the source is not configured");
  });

  it("offers no still-frame link before a frame exists", () => {
    render(<PreviewPanel {...baseProps} hasFrame={false} frameSeq={null} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("links the still frame the ROI editor uses", () => {
    render(<PreviewPanel {...baseProps} />);
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/api/v1/cameras/c1/snapshot.jpg",
    );
  });
});
