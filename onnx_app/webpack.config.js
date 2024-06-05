const path = require("path");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require("copy-webpack-plugin");

const paths = {
    src: path.resolve(__dirname, 'src'),
    dist: path.resolve(__dirname, 'dist')
};

module.exports = {
    context: paths.src,
    mode: "development",

    entry: {
        app: './index'
    },

    output: {
        path: paths.dist,
        filename: '[name].bundle.js',
        publicPath: '/'
    },

    resolve: {
        extensions: ['.ts', '.tsx', '.js', '.jsx'],
        fallback: {
          fs: false,
          path: false,
          crypto: false
        }
    },

    devServer: {
      historyApiFallback: true,
    },

  plugins: [
    new HtmlWebpackPlugin({
        template: './index.html'
    }),
    new CopyPlugin({
      patterns: [
          { from: '../node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
          { from: './best.onnx', to: '[name][ext]'}
      ]
  })
  ],

  module: {
    rules: [
        {
            test: /\.tsx?$/,
            use: [
                {
                    loader: "ts-loader",
                    options: {
                      compilerOptions: {
                        noEmit: false,
                      },
                    },
                },
            ],
        },
      {
        test: /\.css$/,
        use: [
            {
                loader: "style-loader",
            },
          {
            loader: "css-loader",
          }
        ]
      },
      {
        test: /\.(png|svg|jpg|gif)$/,
        use: ["file-loader"]
      }
    ]
  }
};