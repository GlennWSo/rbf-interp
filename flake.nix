{
  description = "Build a cargo project without extra checks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      # Common arguments can be set here to avoid repeating them later
      # Note: changes here will rebuild all dependency crates
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          cargo
          rust-analyzer
          rustfmt
          clippy
        ];
      };
    });
}
