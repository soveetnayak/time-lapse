function output = getLaplacian(input_image, example_input_image, example_output, flags)

  [guidance, laplacian]=direct_method(input_image, example_input_image, example_output, flags);
  output = laplacian\guidance;

  % temp = reshape(output, size(example_output));
  % imwrite(uint8(temp), 'matlab_output.jpg',  'jpg', 'Quality', 100);
end


function [guidance, laplacian]=direct_method(input_image, example_input_image, example_output, flags)

  a_global = getGlobalRegression(example_input_image, example_output);
  
  fprintf('Global matrix:\n');
  disp(a_global)

  patch_size = flags.patch_sz;
  epsilon = flags.epsilon;
  gamma = double(flags.gamma);

  [height, width, num_channel]=size(input_image);
  [~, ~, out_channel]=size(example_output);
  dim_laplacian = numel(input_image)/size(input_image,3);

  %% Column laplacian
  for j = 1 : width - patch_size + 1 
    sparse_column_laplacians{j} = spalloc(dim_laplacian, dim_laplacian, height*patch_size*(2*patch_size-1)^2);
  end

  column_guidances=zeros(dim_laplacian, out_channel, width - patch_size + 1 );

  %% Image laplacian
  laplacian = spalloc(dim_laplacian, dim_laplacian, height*width*patch_size^2);

  %% Local laplacian
  for j = 1 : width - patch_size + 1  
    local_laplacians=zeros(patch_size^2, patch_size^2, width - patch_size + 1);
    local_guidances =zeros(patch_size^2, out_channel, width - patch_size + 1);
    global_index=zeros(patch_size^2, height - patch_size + 1);

    XX=zeros(patch_size^2, num_channel, height-patch_size+1);
    XX_0=zeros(patch_size^2, num_channel, height-patch_size+1);
    YY_0=zeros(patch_size^2, out_channel, height-patch_size+1);

    for i = 1 : height - patch_size + 1 
      x_patch=input_image(i:i+patch_size-1, j:j+patch_size-1,:);
      x_0_patch=example_input_image(i:i+patch_size-1, j:j+patch_size-1,:);
      y_0_patch=example_output(i:i+patch_size-1, j:j+patch_size-1,:);
      XX(:,:,i)=reshape(x_patch, [patch_size^2, num_channel]);
      XX_0(:,:,i)=reshape(x_0_patch, [patch_size^2, num_channel]);
      YY_0(:,:,i)=reshape(y_0_patch, [patch_size^2, out_channel]);
    end
    
    for i = 1 : height - patch_size + 1 
      local_laplacians(:,:,i) = LocalLaplacian(XX(:,:,i), XX_0(:,:,i), epsilon, gamma);
      global_index(:,i) = FastGlobalIndex(i:i + patch_size-1, 1:patch_size, height, patch_size);
      local_guidances(:,:,i) = LocalGuidance(XX(:,:,i), XX_0(:,:,i), YY_0(:,:,i), epsilon, gamma, a_global);
    end

    column_laplacian=zeros(height*patch_size);
    column_guidance=zeros(height*patch_size, out_channel);
    
    for i= 1 : height - patch_size + 1
      elements=global_index(:,i);
      column_laplacian(elements, elements) = column_laplacian(elements, elements) + local_laplacians(:,:,i);
      column_guidance(elements,:) = column_guidance(elements,:) + local_guidances(:,:,i);
    end
    
    ul_point = (j-1)*height + 1;
    br_point = (j-1)*height + height*patch_size;
    sparse_column_laplacians{j}(ul_point:br_point, ul_point:br_point) = column_laplacian;

    foo=zeros(dim_laplacian, out_channel);
    foo(ul_point:br_point, :)=column_guidance;
    column_guidances(:, :, j)=foo;
  end
  
  laplacian = sparse_sum(sparse_column_laplacians);
  guidance  = sum(column_guidances, 3);
end

%% Helpers for direct method
function a_global=getGlobalRegression(input, output)

  [h_e, w_e, num_channel_e] = size(output);
  [h_i, w_i, num_channel_i] = size(input);

  example_input_vec    = reshape(input, h_i*w_i, num_channel_i);
  example_output_vec   = reshape(output, h_e*w_e, num_channel_e);

  sigma = example_input_vec'*example_input_vec;
  a_global= inv(sigma)*(example_input_vec'*example_output_vec);
end

function out = sparse_sum(sparse_mats)
  if length(sparse_mats) == 2
    out =  sparse_mats{1} + sparse_mats{2};
  elseif length(sparse_mats)==1
    out =  sparse_mats{1};
  else
    queue_length = length(sparse_mats);
    left=floor(queue_length/2);
    out = sparse_sum({sparse_mats{1:left}}) + sparse_sum({sparse_mats{left+1:queue_length}});
  end
end

function global_index = FastGlobalIndex(i_slice, j_slice, im_height, im_width)

    global_sub_i = repmat(i_slice(:), [length(i_slice) 1]);
    global_sub_j = repmat(j_slice(:)', [length(i_slice) 1]);
    global_index = sub2ind([im_height im_width], global_sub_i(:), global_sub_j(:));
end 

function local_laplacian = LocalLaplacian(x, x_0, epsilon, gamma)
  num_channel = size(x,2);
  dim_local_laplacian = numel(x)/num_channel;
  precision_matrix = inv((x')*x + epsilon*(x_0')*x_0 + gamma*eye(num_channel));
  local_laplacian = eye(dim_local_laplacian) - x*precision_matrix*x';
end

function local_guidance = LocalGuidance(x, x_0, y_0, epsilon, gamma, a_global)
  num_channel = size(x,2);
  precision_matrix=inv((x')*x + epsilon*(x_0')*x_0 + gamma*eye(num_channel));
  local_guidance = x*precision_matrix*(epsilon*x_0'*y_0 + gamma*a_global);
end