%global app_id app.sculptlite.editor
%global debug_package %{nil}

Name:           sculpt-lite
Version:        0.1.0
Release:        1%{?dist}
Summary:        Native focused STL mesh sculpting tool
License:        LicenseRef-Proprietary
URL:            https://github.com/kaufmann-dev/sculpt-lite
Source0:        sculpt-lite
Source1:        %{app_id}.desktop
Source2:        %{app_id}.metainfo.xml
Source3:        %{app_id}.png
ExclusiveArch:  x86_64

BuildRequires:  appstream
BuildRequires:  desktop-file-utils
BuildRequires:  libxml2
Requires:       libX11
Requires:       libXcursor
Requires:       libXi
Requires:       libXrandr
Requires:       libglvnd-egl
Requires:       libglvnd-gles
Requires:       libwayland-client
Requires:       libxkbcommon
Requires:       libxkbcommon-x11
Requires:       vulkan-loader

%description
SculptLite is a native Linux tool for focused, organic sculpting of STL meshes.
It imports STL files, provides direct fixed-topology sculpting brushes, and
exports the edited result as STL.

%prep

%build

%install
install -Dpm0755 %{SOURCE0} %{buildroot}%{_bindir}/sculpt-lite
install -Dpm0644 %{SOURCE1} %{buildroot}%{_datadir}/applications/%{app_id}.desktop
install -Dpm0644 %{SOURCE2} %{buildroot}%{_metainfodir}/%{app_id}.metainfo.xml
install -Dpm0644 %{SOURCE3} %{buildroot}%{_datadir}/icons/hicolor/512x512/apps/%{app_id}.png

%check
desktop-file-validate %{SOURCE1}
appstreamcli validate --no-net %{SOURCE2}
xmllint --noout %{SOURCE2}

%files
%{_bindir}/sculpt-lite
%{_datadir}/applications/%{app_id}.desktop
%{_metainfodir}/%{app_id}.metainfo.xml
%{_datadir}/icons/hicolor/512x512/apps/%{app_id}.png

%changelog
* Sat Jul 18 2026 SculptLite maintainers <maintainers@sculptlite.invalid> - 0.1.0-1
- Build the first native Linux package
