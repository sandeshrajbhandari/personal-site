module.exports = {
  images: {
    loader: 'custom',
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    nextImageExportOptimizer: {
      imageFolderPath: 'public',
      exportFolderPath: 'out',
      quality: 75
    },
    domains: [
      'i.scdn.co', // Spotify Album Art
      'pbs.twimg.com' // Twitter Profile Picture
    ]
  },

  env: {
    storePicturesInWEBP: false,
    generateAndUseBlurImages: true
  },

  swcMinify: true,
  reactStrictMode: true,
  webpack: (config, { dev, isServer }) => {
    // Replace React with Preact only in client production build
    if (!dev && !isServer) {
      Object.assign(config.resolve.alias, {
        react: 'preact/compat',
        'react-dom/test-utils': 'preact/test-utils',
        'react-dom': 'preact/compat'
      });
    }

    return config;
  }
};