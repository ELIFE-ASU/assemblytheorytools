import matplotlib

# Several tests call plt.show() on real figures. Without a non-interactive
# backend, plt.show() blocks waiting for a display that never appears in a
# headless/CI environment, hanging the whole test run indefinitely instead
# of failing fast. Setting the Agg backend here, before any test module
# imports pyplot, turns plt.show() into a harmless no-op.
matplotlib.use("Agg")
