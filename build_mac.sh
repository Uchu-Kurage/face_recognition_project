#!/bin/bash

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -rf build dist

# Run PyInstaller
echo "Running PyInstaller..."
python3 -m PyInstaller Omokage.spec

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "App is located at dist/Omokage.app"
else
    echo "Build failed!"
    exit 1
fi
