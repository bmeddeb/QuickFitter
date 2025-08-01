# QuickFitter

An interactive web-based tool for fitting dielectric spectroscopy data using advanced models. QuickFitter provides a user-friendly interface for researchers and engineers to analyze dielectric properties of materials with real-time parameter adjustment and comprehensive fit evaluation.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

## Features

- **Interactive Data Fitting**: Real-time parameter adjustment with immediate visual feedback
- **Djordjevic-Sarkar Model**: Implementation of the Djordjevic Sarkar model for material characterization
- **Fit Quality Evaluation**: Comprehensive metrics to assess fit quality with visual indicators
- **Multiple Export Options**: Download results as a zip for CSV, PNG plots, JSON reports, and evaluation metrics
- **Browser-Based**: No installation required - runs entirely in your web browser using Pyodide
- **Responsive Design**: Works on desktop and tablet devices

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/bmeddeb/QuickFitter.git
cd QuickFitter
```

2. Serve the application locally:
```bash
python -m http.server 8000
```
OR double click on index.html

3. Open your browser and navigate to `http://localhost:8000`

4. Upload your CSV file with columns: `Frequency (GHz)`, `Dk`, `Df`

## Input Data Format

Your CSV file should have three columns:
- **Frequency (GHz)**: Measurement frequency in gigahertz
- **Dk**: Dielectric constant (real part of permittivity)
- **Df**: Dissipation factor (loss tangent)

Example:
```csv
Frequency (GHz),Dk,Df
1.0,3.45,0.002
2.0,3.42,0.003
5.0,3.38,0.005
```

## Models

### Currently Implemented
- **Djordjevic-Sarkar Model**: A wideband model for frequency-dependent dielectric materials
  - Parameters: �, ��, f�, f�
  - Suitable for: Materials with frequency-dependent permittivity and loss

### Coming Soon
- **Debye Model**: For materials with single relaxation time
- **Cole-Cole Model**: For materials with distributed relaxation times
- **Havriliak-Negami Model**: For asymmetric relaxation processes
- **Multi-pole Models**: For complex material systems
- **Custom Model Support**: Upload your own model equations

## Fit Quality Metrics

QuickFitter evaluates your fit using multiple metrics:
- **Reduced Chi-Square**: Overall goodness of fit
- **RMS Errors**: For real and imaginary parts separately
- **Parameter Correlations**: To identify parameter dependencies
- **Data Points per Variable**: To ensure statistical validity

## Output Files

The tool generates a comprehensive results package:
- `*_fitted.csv`: Fitted data points
- `*_plot.png`: Publication-ready plot
- `*_report.txt`: Detailed fit report
- `*_report.json`: Machine-readable results
- `*_evaluation.md`: Fit quality assessment

## Technology Stack

- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **Backend**: Python (via Pyodide in browser)
- **Scientific Libraries**: NumPy, SciPy, lmfit, Matplotlib, Plotly
- **Build**: No build process required - runs directly in browser

## Citation

If you use QuickFitter in your research, please cite:

```bibtex
@software{quickfitter2025,
  author = {Meddeb, Ben and Meddeb, Amira},
  title = {QuickFitter: Interactive Dielectric Data Fitting Tool},
  year = {2025},
  url = {https://github.com/bmeddeb/QuickFitter}
}
```

## Developers

- **Ben Meddeb** - Software Engineer
- **Amira Meddeb** - Material Engineer

## Contributing

We welcome contributions! Areas where you can help:
- Adding new dielectric models
- Improving the fitting algorithms
- Enhancing the user interface
- Adding data preprocessing features
- Writing documentation and tutorials

Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Djordjevic and Sarkar for their groundbreaking dielectric model
- The Pyodide team for enabling Python in the browser
- The scientific Python community for the excellent libraries

## Roadmap

### Upcoming Features
- [ ] Multiple model comparison
- [ ] Batch processing capabilities
- [ ] Data preprocessing tools
- [ ] Uncertainty analysis
- [ ] Export to common simulation formats
- [ ] Temperature-dependent fitting
- [ ] Multi-dataset simultaneous fitting

### Future Models
- [ ] Debye relaxation
- [ ] Cole-Cole distribution
- [ ] Havriliak-Negami model
- [ ] Power law models
- [ ] Custom equation support

## Support

For questions, bug reports, or feature requests, please:
- Open an issue on [GitHub](https://github.com/bmeddeb/QuickFitter/issues)
- Contact the developers through the repository

---

Made with d for the materials science community