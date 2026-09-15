"""Reject empty, skipped, or unsuccessful required pytest JUnit reports."""

import argparse
import xml.etree.ElementTree as ET


def check_report(path):
    """Require at least one executed test and no skipped or failed test cases."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("The test report contains no executed test cases")
    if root.find(".//skipped") is not None:
        raise ValueError("The required lane must not skip tests")
    if root.find(".//failure") is not None or root.find(".//error") is not None:
        raise ValueError("The test report contains failures or errors")
    print(f"Verified {len(cases)} test cases with no skips or failures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    check_report(parser.parse_args().report)
